"""Gap-closure report: how much of the teacher/student gap each distillation arm closed.

gap_closure(arm) = (distilled_acc - base_acc) / (teacher_acc - base_acc)

Combines the five eval.py reports (base student, teacher, novelty-search-distilled,
ground-truth-distilled, NeMo-baseline-distilled) into a single comparison, answering whether
novelty-search-generated synthetic data closes more of the gap than plain ground-truth GSM8K data
or a flat NeMo Data Designer synthetic baseline.

Usage:
    PYTHONPATH=. python examples/gsm8k_distillation/report.py \
        --base-report output/eval/base/report.json \
        --teacher-report output/eval/teacher/report.json \
        --synthetic-report output/eval/synthetic/report.json \
        --groundtruth-report output/eval/groundtruth/report.json \
        --nemo-report output/eval/nemo_baseline/report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from examples.gsm8k_distillation.common.files import read_json, write_json
from examples.gsm8k_distillation.common.logging_utils import get_logger
from examples.gsm8k_distillation.common.paths import OUTPUT_DIR
from examples.gsm8k_distillation.common.plots import save_figure

logger = get_logger(__name__)

METRIC = "mean_correctness_rate"


def gap_closure(distilled_acc: float, base_acc: float, teacher_acc: float) -> float:
    denom = teacher_acc - base_acc
    if denom == 0:
        return 0.0
    return (distilled_acc - base_acc) / denom


def build_gap_closure_report(base: dict, teacher: dict, synthetic: dict, groundtruth: dict, nemo: dict) -> dict:
    base_acc = base[METRIC]
    teacher_acc = teacher[METRIC]
    synthetic_acc = synthetic[METRIC]
    groundtruth_acc = groundtruth[METRIC]
    nemo_acc = nemo[METRIC]

    return {
        "metric": METRIC,
        "base_accuracy": base_acc,
        "teacher_accuracy": teacher_acc,
        "synthetic_accuracy": synthetic_acc,
        "groundtruth_accuracy": groundtruth_acc,
        "nemo_accuracy": nemo_acc,
        "synthetic_gap_closure": gap_closure(synthetic_acc, base_acc, teacher_acc),
        "groundtruth_gap_closure": gap_closure(groundtruth_acc, base_acc, teacher_acc),
        "nemo_gap_closure": gap_closure(nemo_acc, base_acc, teacher_acc),
    }


def print_summary(report: dict) -> None:
    lines = [
        "",
        f"{'arm':<14}{'accuracy':>10}{'gap closure':>14}",
        f"{'base':<14}{report['base_accuracy']:>10.3f}{'-':>14}",
        f"{'synthetic':<14}{report['synthetic_accuracy']:>10.3f}{report['synthetic_gap_closure']:>14.1%}",
        f"{'groundtruth':<14}{report['groundtruth_accuracy']:>10.3f}{report['groundtruth_gap_closure']:>14.1%}",
        f"{'nemo':<14}{report['nemo_accuracy']:>10.3f}{report['nemo_gap_closure']:>14.1%}",
        f"{'teacher':<14}{report['teacher_accuracy']:>10.3f}{'-':>14}",
        "",
    ]
    arms = {
        "synthetic": report["synthetic_gap_closure"],
        "groundtruth": report["groundtruth_gap_closure"],
        "nemo": report["nemo_gap_closure"],
    }
    best = max(arms, key=arms.get)
    if len({round(v, 6) for v in arms.values()}) == 1:
        lines.append("Synthetic, ground-truth, and NeMo baseline data closed the gap equally.")
    else:
        lines.append(f"{best.capitalize()} data closed the most of the teacher/student gap ({arms[best]:.1%}).")
    print("\n".join(lines))


def plot_gap_closure(report: dict, output_path: Path) -> None:
    labels = ["base", "synthetic", "groundtruth", "nemo", "teacher"]
    values = [
        report["base_accuracy"],
        report["synthetic_accuracy"],
        report["groundtruth_accuracy"],
        report["nemo_accuracy"],
        report["teacher_accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel(METRIC)
    ax.set_title("GSM8K val accuracy: base vs distilled arms vs teacher")

    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--synthetic-report", type=Path, required=True)
    parser.add_argument("--groundtruth-report", type=Path, required=True)
    parser.add_argument("--nemo-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "gap_closure_report.json")
    args = parser.parse_args()

    base = read_json(args.base_report)
    teacher = read_json(args.teacher_report)
    synthetic = read_json(args.synthetic_report)
    groundtruth = read_json(args.groundtruth_report)
    nemo = read_json(args.nemo_report)

    report = build_gap_closure_report(base, teacher, synthetic, groundtruth, nemo)

    write_json(report, args.output)
    logger.info("Wrote gap-closure report to %s", args.output)

    plot_gap_closure(report, args.output.parent / "gap_closure_report.png")
    logger.info("Wrote gap-closure plot to %s", args.output.parent / "gap_closure_report.png")

    print_summary(report)


if __name__ == "__main__":
    main()
