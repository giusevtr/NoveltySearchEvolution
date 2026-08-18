"""Diversity/coverage comparison: ground-truth GSM8K questions vs. both synthetic arms.

Compares novelty search's output/synthetic_distillation.parquet and the NeMo Data Designer
baseline's output/baseline_nemo_distillation.parquet against real GSM8K questions (annotated
with teacher completions) along two components:
  - "prompt": the question text itself.
  - "teacher_annotation": the teacher model's completion/reasoning for that question.
For each component this produces:
  - Quantitative metrics (output/diversity_metrics.json): average pairwise cosine distance within
    each dataset (diversity) and, for each synthetic set, the mean cosine distance from each
    ground-truth embedding to its nearest synthetic neighbor (coverage — how well the synthetic
    set covers the ground-truth distribution).
  - A t-SNE panel (output/diversity_tsne.png, two panels side by side) with all three datasets
    overlaid.
Text is embedded with the same semantic embedding novelty search itself uses for selection
(novelty_search/embedding.py::question_embedding_fn, Bedrock Titan text embeddings), and distances
use cosine, matching run_novelty_search_augmentatin.py's own SELECTION_CONFIG distance metric — so "diversity" here means the
same thing it means to the novelty search engine.

Usage:
    PYTHONPATH=. python examples/gsm8k_distillation/diversity_report.py [--smoke-test]

Requires AWS Bedrock credentials (standard boto3 chain), and
output/synthetic_distillation.parquet (produced by run_novelty_search_augmentatin.py),
output/baseline_nemo_distillation.parquet (produced by baseline_nemo.py), and
data/gsm8k_train_teacher_annotated.parquet (produced by scripts/annotate_groundtruth.py) to
already exist.
Embeddings are cached under output/embedding_cache/ and reused across runs when the underlying
text set is unchanged, so repeat runs skip Bedrock calls entirely.
"""

from __future__ import annotations

import sys
import argparse
import hashlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from examples.gsm8k_distillation.common.files import load_frame, require_file, write_json
from examples.gsm8k_distillation.common.logging_utils import get_logger
from examples.gsm8k_distillation.common.paths import DATA_DIR, OUTPUT_DIR
from examples.gsm8k_distillation.common.plots import save_figure
from examples.gsm8k_distillation.novelty_search.embedding import question_embedding_fn
from novelty_search_evolution import average_pairwise_distance, mean_min_distance_to_reference, vendi_score

logger = get_logger(__name__)

DEFAULT_SYNTHETIC_DATA = OUTPUT_DIR / "synthetic_distillation.parquet"
DEFAULT_NEMO_DATA = OUTPUT_DIR / "baseline_nemo_distillation.parquet"
DEFAULT_GROUNDTRUTH_DATA = DATA_DIR / "gsm8k_train_teacher_annotated.parquet"
DEFAULT_OUTPUT = OUTPUT_DIR / "diversity_tsne.png"
DEFAULT_METRICS_OUTPUT = OUTPUT_DIR / "diversity_metrics.json"
DEFAULT_CACHE_DIR = OUTPUT_DIR / "embedding_cache"

SMOKE_TEST_NUM_SAMPLES = 20

COMPONENTS = ("prompt", "teacher_annotation")
_COMPONENT_COLUMNS = {"prompt": "question", "teacher_annotation": "teacher_completion"}


def embed(questions: list[str]) -> np.ndarray:
    return question_embedding_fn(questions)


def _cache_key(questions: list[str]) -> str:
    return hashlib.sha256("\n".join(questions).encode("utf-8")).hexdigest()[:16]


def embed_cached(questions: list[str], tag: str, cache_dir: Path, refresh: bool = False) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{tag}_{_cache_key(questions)}.npz"
    if not refresh and cache_path.exists():
        # allow_pickle stays off: a cache file is just arrays, and unpickling one would execute
        # whatever code it carries.
        try:
            with np.load(cache_path) as cached:
                cached_questions = cached["questions"].tolist()
                cached_embeddings = cached["embeddings"]
        except ValueError:
            logger.warning("Ignoring unreadable %s embedding cache at %s; regenerating", tag, cache_path)
        else:
            if cached_questions == questions:
                logger.info("Loaded cached %s embeddings (%d questions) from %s", tag, len(questions), cache_path)
                return cached_embeddings

    embeddings = embed(questions)
    np.savez(cache_path, questions=np.array(questions, dtype=np.str_), embeddings=embeddings)
    logger.info("Cached %s embeddings to %s", tag, cache_path)
    return embeddings


def embed_components(df: pd.DataFrame, tag: str, cache_dir: Path, refresh: bool = False) -> dict[str, np.ndarray]:
    """Embed each component's text column, returning {component: (N, D) embeddings}."""
    return {
        component: embed_cached(df[column].tolist(), tag=f"{tag}_{component}", cache_dir=cache_dir, refresh=refresh)
        for component, column in _COMPONENT_COLUMNS.items()
    }


def _dataset_size(path: Path) -> int:
    return len(load_frame(path))


def compute_tsne_2d(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 3:
        raise ValueError(f"t-SNE needs at least 3 points, got {len(embeddings)}")
    # sklearn requires perplexity < n_samples
    perplexity = min(30, max(1, (len(embeddings) - 1) // 3), len(embeddings) - 1)
    return TSNE(n_components=2, random_state=0, perplexity=perplexity).fit_transform(embeddings)


def compute_component_metrics(gt_embeddings: np.ndarray, novelty_embeddings: np.ndarray, nemo_embeddings: np.ndarray) -> dict:
    return {
        "groundtruth": {
            "n": len(gt_embeddings),
            "avg_pairwise_distance": average_pairwise_distance(gt_embeddings),
            "vendi_score": vendi_score(gt_embeddings),
        },
        "novelty_search": {
            "n": len(novelty_embeddings),
            "avg_pairwise_distance": average_pairwise_distance(novelty_embeddings),
            "vendi_score": vendi_score(novelty_embeddings),
            "coverage_distance_to_groundtruth": mean_min_distance_to_reference(novelty_embeddings, gt_embeddings),
            "coverage_distance_from_groundtruth": mean_min_distance_to_reference(gt_embeddings, novelty_embeddings),
        },
        "nemo_baseline": {
            "n": len(nemo_embeddings),
            "avg_pairwise_distance": average_pairwise_distance(nemo_embeddings),
            "vendi_score": vendi_score(nemo_embeddings),
            "coverage_distance_to_groundtruth": mean_min_distance_to_reference(nemo_embeddings, gt_embeddings),
            "coverage_distance_from_groundtruth": mean_min_distance_to_reference(gt_embeddings, nemo_embeddings),
        },
    }


def compute_metrics(
    gt_embeds: dict[str, np.ndarray], novelty_embeds: dict[str, np.ndarray], nemo_embeds: dict[str, np.ndarray]
) -> dict:
    return {
        component: compute_component_metrics(gt_embeds[component], novelty_embeds[component], nemo_embeds[component])
        for component in COMPONENTS
    }


def print_component_metrics_summary(component: str, component_metrics: dict) -> None:
    gt, novelty, nemo = component_metrics["groundtruth"], component_metrics["novelty_search"], component_metrics["nemo_baseline"]
    lines = [
        "",
        f"--- {component} ---",
        f"{'dataset':<16}{'n':>6}{'avg pairwise dist':>20}{'vendi':>10}{'coverage dist to GT':>22}{'coverage dist from GT':>24}",
        f"{'groundtruth':<16}{gt['n']:>6}{gt['avg_pairwise_distance']:>20.4f}{gt['vendi_score']:>10.2f}{'-':>22}{'-':>24}",
        f"{'novelty_search':<16}{novelty['n']:>6}{novelty['avg_pairwise_distance']:>20.4f}{novelty['vendi_score']:>10.2f}{novelty['coverage_distance_to_groundtruth']:>22.4f}{novelty['coverage_distance_from_groundtruth']:>24.4f}",
        f"{'nemo_baseline':<16}{nemo['n']:>6}{nemo['avg_pairwise_distance']:>20.4f}{nemo['vendi_score']:>10.2f}{nemo['coverage_distance_to_groundtruth']:>22.4f}{nemo['coverage_distance_from_groundtruth']:>24.4f}",
        "",
    ]
    if novelty["avg_pairwise_distance"] > nemo["avg_pairwise_distance"]:
        lines.append("Novelty search is more internally diverse than the NeMo baseline.")
    elif nemo["avg_pairwise_distance"] > novelty["avg_pairwise_distance"]:
        lines.append("The NeMo baseline is more internally diverse than novelty search.")
    else:
        lines.append("Novelty search and the NeMo baseline are equally diverse.")
    if novelty["vendi_score"] > nemo["vendi_score"]:
        lines.append("Novelty search has a higher Vendi score (more effective distinct samples).")
    elif nemo["vendi_score"] > novelty["vendi_score"]:
        lines.append("The NeMo baseline has a higher Vendi score (more effective distinct samples).")
    else:
        lines.append("Novelty search and the NeMo baseline have equal Vendi scores.")
    if novelty["coverage_distance_to_groundtruth"] < nemo["coverage_distance_to_groundtruth"]:
        lines.append("Novelty search has lower mean min-distance to ground truth (better coverage).")
    elif nemo["coverage_distance_to_groundtruth"] < novelty["coverage_distance_to_groundtruth"]:
        lines.append("The NeMo baseline has lower mean min-distance to ground truth (better coverage).")
    else:
        lines.append("Novelty search and the NeMo baseline cover ground truth equally well.")
    if novelty["coverage_distance_from_groundtruth"] < nemo["coverage_distance_from_groundtruth"]:
        lines.append("Novelty search stays closer to the ground-truth manifold (better precision).")
    elif nemo["coverage_distance_from_groundtruth"] < novelty["coverage_distance_from_groundtruth"]:
        lines.append("The NeMo baseline stays closer to the ground-truth manifold (better precision).")
    else:
        lines.append("Novelty search and the NeMo baseline are equally close to the ground-truth manifold.")
    print("\n".join(lines))


def print_metrics_summary(metrics: dict) -> None:
    for component in COMPONENTS:
        print_component_metrics_summary(component, metrics[component])


def plot_component(ax, gt_2d: np.ndarray, novelty_2d: np.ndarray, nemo_2d: np.ndarray, title: str) -> None:
    ax.scatter(gt_2d[:, 0], gt_2d[:, 1], c="gray", alpha=0.4, s=15, label="ground truth")
    ax.scatter(novelty_2d[:, 0], novelty_2d[:, 1], c="tab:blue", alpha=0.6, s=15, label="novelty search")
    ax.scatter(nemo_2d[:, 0], nemo_2d[:, 1], c="tab:orange", alpha=0.6, s=15, label="nemo baseline")
    ax.set_title(title)
    ax.legend()


def plot_combined(
    tsne_2d_by_component: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(8 * len(COMPONENTS), 7))
    titles = {"prompt": "Prompt Diversity (t-SNE)", "teacher_annotation": "Teacher Annotation Diversity (t-SNE)"}
    for ax, component in zip(axes, COMPONENTS):
        gt_2d, novelty_2d, nemo_2d = tsne_2d_by_component[component]
        plot_component(ax, gt_2d, novelty_2d, nemo_2d, titles[component])
    fig.suptitle("Synthetic Question Diversity vs. Ground Truth (t-SNE of semantic embeddings)")
    fig.tight_layout()

    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-data", type=Path, default=DEFAULT_SYNTHETIC_DATA)
    parser.add_argument("--nemo-data", type=Path, default=DEFAULT_NEMO_DATA)
    parser.add_argument("--groundtruth-data", type=Path, default=DEFAULT_GROUNDTRUTH_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute embeddings even if a matching cache entry exists.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Cap ground-truth and synthetic sample counts for a fast end-to-end wiring check.",
    )
    args = parser.parse_args()

    require_file(args.synthetic_data, "run run_novelty_search_augmentatin.py first to generate it.")
    require_file(args.nemo_data, "run baseline_nemo.py first to generate it.")
    require_file(args.groundtruth_data, "run scripts/annotate_groundtruth.py first to generate it.")

    if args.smoke_test:
        num_samples = SMOKE_TEST_NUM_SAMPLES
    else:
        dataset_sizes = {
            "groundtruth": _dataset_size(args.groundtruth_data),
            "novelty_search": _dataset_size(args.synthetic_data),
            "nemo_baseline": _dataset_size(args.nemo_data),
        }
        num_samples = min(dataset_sizes.values())
        logger.info("Dataset sizes %s; capping all three to %d samples", dataset_sizes, num_samples)

    gt_df = load_frame(args.groundtruth_data, n=num_samples)
    logger.info("Loaded %d ground-truth rows", len(gt_df))
    gt_embeds = embed_components(gt_df, tag="groundtruth", cache_dir=args.cache_dir, refresh=args.refresh_cache)

    novelty_df = load_frame(args.synthetic_data, n=num_samples)
    logger.info("Loaded %d novelty-search synthetic rows", len(novelty_df))
    novelty_embeds = embed_components(novelty_df, tag="novelty", cache_dir=args.cache_dir, refresh=args.refresh_cache)

    nemo_df = load_frame(args.nemo_data, n=num_samples)
    logger.info("Loaded %d NeMo baseline synthetic rows", len(nemo_df))
    nemo_embeds = embed_components(nemo_df, tag="nemo", cache_dir=args.cache_dir, refresh=args.refresh_cache)

    metrics = compute_metrics(gt_embeds, novelty_embeds, nemo_embeds)
    write_json(metrics, args.metrics_output)
    logger.info("Wrote diversity metrics to %s", args.metrics_output)
    print_metrics_summary(metrics)

    tsne_2d_by_component = {}
    for component in COMPONENTS:
        gt_embeddings, novelty_embeddings, nemo_embeddings = (
            gt_embeds[component],
            novelty_embeds[component],
            nemo_embeds[component],
        )
        n_gt, n_novelty = len(gt_embeddings), len(novelty_embeddings)
        combined = np.concatenate([gt_embeddings, novelty_embeddings, nemo_embeddings], axis=0)
        tsne_2d = compute_tsne_2d(combined)
        tsne_2d_by_component[component] = (
            tsne_2d[:n_gt],
            tsne_2d[n_gt : n_gt + n_novelty],
            tsne_2d[n_gt + n_novelty :],
        )

    plot_combined(tsne_2d_by_component, args.output)
    logger.info("Wrote diversity t-SNE plot to %s", args.output)


if __name__ == "__main__":
    main()
