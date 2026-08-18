"""GSM8K distillation data augmentation via novelty search.

Mutates 5 seed GSM8K questions using a Bedrock-hosted Qwen3-32B teacher, generates teacher
completions (ground truth), scores each question's difficulty by sampling K=10 completions
from a colocated vLLM Qwen3-1.7B student and counting how many match the teacher's answer,
embeds each sample as a concatenation of unit-norm question semantics (Bedrock Titan text
embeddings) and a unit-norm one-hot difficulty encoding, filters out samples with malformed
teacher completions, and
grows the population via `novelty_search_evolution`'s novelty search engine until NUM_SAMPLES
samples have been accepted.

Produces `output/synthetic_distillation.parquet` and a difficulty-distribution plot.

Usage:
    PYTHONPATH=. python examples/gsm8k_distillation/scripts/run_novelty_search_augmentatin.py [--smoke-test]

Requires AWS Bedrock credentials (standard boto3 chain) for the Qwen3-32B teacher, and a
CUDA GPU for the colocated Qwen3-1.7B vLLM student.
"""

from __future__ import annotations
import sys
import argparse
import random
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd

from examples.gsm8k_distillation.clients.student_client import StudentClient, StudentConfig
from examples.gsm8k_distillation.clients.teacher_client import TeacherClient
from examples.gsm8k_distillation.common.files import load_seed_questions
from examples.gsm8k_distillation.common.inference import (
    K_STUDENT_SAMPLES,
    solve_and_score,
    teacher_write_questions,
)
from examples.gsm8k_distillation.common.logging_utils import get_logger
from examples.gsm8k_distillation.common.paths import GSM8K_TRAIN_PARQUET, LOG_DIR, OUTPUT_DIR
from examples.gsm8k_distillation.common.plots import plot_difficulty_histogram
from examples.gsm8k_distillation.novelty_search.embedding import combine_embeddings, question_embedding_fn
from examples.gsm8k_distillation.novelty_search.prompts import (
    TOPICS,
    build_crossover_prompt,
    build_mutation_prompt,
    extract_gsm8k_answer,
)

from novelty_search_evolution import EvolutionEngine, EvoSample, Population

logger = get_logger(__name__)

# --- config ---

NUM_SEEDS = 5
NUM_SAMPLES = 8000
NOVELTY_KNN_K = 10
MAX_GENERATIONS_SAFETY_CAP = 300

MUTATIONS_PER_PARENT = 5
NUM_MUTATION_SAMPLES = 10  # parents sampled per generation
CROSSOVER_CHILDREN_PER_PAIR = 5
NUM_CROSSOVER_SAMPLES = 5  # parent pairs sampled per generation
SELECTION_SIZE = 15

SELECTION_CONFIG = {
    "nn-k": NOVELTY_KNN_K,
    "distance": "cosine",
    "archive_update_prob": 0.3,
}

CHECKPOINT_EVERY_N_GENERATIONS = 10
SYNTHETIC_DATA_PATH = OUTPUT_DIR / "synthetic_distillation.parquet"
DIFFICULTY_PLOT_PATH = OUTPUT_DIR / "difficulty_distribution.png"
DEFAULT_WANDB_PROJECT = "gsm8k-distillation-novelty-search"


# --- data assembly ---


def _assemble_dict(
    question: str,
    teacher_completion: str,
    difficulty: int,
    source: str,
    parent_question: str | None,
    topic: str | None,
    parent_question_2: str | None = None,
) -> dict:
    return {
        "question": question,
        "teacher_completion": teacher_completion,
        "final_answer": extract_gsm8k_answer(teacher_completion),
        "difficulty": difficulty,
        "source": source,
        "parent_question": parent_question,
        "parent_question_2": parent_question_2,
        "topic": topic,
    }


def load_seeds(teacher: TeacherClient, student: StudentClient, n: int, random_state: int = 0) -> list[dict]:
    seed_questions = load_seed_questions(GSM8K_TRAIN_PARQUET, n=n, random_state=random_state)
    teacher_completions, difficulties = solve_and_score(teacher, student, seed_questions)

    return [
        _assemble_dict(q, c, d, source="seed", parent_question=None, topic=None)
        for q, c, d in zip(seed_questions, teacher_completions, difficulties)
    ]


# --- evolution operators ---


def _breed(
    teacher: TeacherClient,
    student: StudentClient,
    build_prompt: Callable[[str], str],
    topics: list[str],
    source: str,
    parent_question: str,
    parent_question_2: str | None = None,
) -> list[dict]:
    """Ask the teacher for one new question per topic, then solve and difficulty-score each."""
    new_questions = teacher_write_questions(teacher, [build_prompt(topic) for topic in topics])
    teacher_completions, difficulties = solve_and_score(teacher, student, new_questions)

    return [
        _assemble_dict(
            q,
            c,
            d,
            source=source,
            parent_question=parent_question,
            parent_question_2=parent_question_2,
            topic=topic,
        )
        for q, c, d, topic in zip(new_questions, teacher_completions, difficulties, topics)
    ]


def _random_topics(n: int) -> list[str]:
    return [random.choice(TOPICS) for _ in range(n)]


def make_mut_fn(teacher: TeacherClient, student: StudentClient, mutations_per_parent: int):
    def mut_fn(parent: EvoSample) -> list[dict]:
        parent_q = parent.get_data()["question"]
        return _breed(
            teacher,
            student,
            lambda topic: build_mutation_prompt(parent_q, topic),
            _random_topics(mutations_per_parent),
            source="mutation",
            parent_question=parent_q,
        )

    return mut_fn


def make_crossover_fn(teacher: TeacherClient, student: StudentClient, children_per_pair: int):
    def crossover_fn(parent_1: EvoSample, parent_2: EvoSample) -> list[dict]:
        q1 = parent_1.get_data()["question"]
        q2 = parent_2.get_data()["question"]
        return _breed(
            teacher,
            student,
            lambda topic: build_crossover_prompt(q1, q2, topic),
            _random_topics(children_per_pair),
            source="crossover",
            parent_question=q1,
            parent_question_2=q2,
        )

    return crossover_fn


def format_filter_fn(candidates: list[EvoSample]) -> list[dict]:
    judgements = []
    for c in candidates:
        d = c.get_data()
        ok = d["final_answer"] is not None
        judgements.append({"accepted": ok, "feedback": "" if ok else "missing/invalid #### <number> terminator"})
    return judgements


def embedding_fn(data_list: list[dict]) -> list[np.ndarray]:
    questions = [d["question"] for d in data_list]
    completions = [d["teacher_completion"] for d in data_list]
    difficulties = [d["difficulty"] for d in data_list]
    sem_vecs = question_embedding_fn(questions)
    completion_vecs = question_embedding_fn(completions)
    return combine_embeddings(sem_vecs, completion_vecs, difficulties, K_STUDENT_SAMPLES)


# --- stopping condition ---


def kept_samples(population: Population) -> list[EvoSample]:
    """All non-rejected samples — anything that passed every filter, whether or not it was
    ever novelty-selected into the active pool or archived."""
    return population.get_accepted()


def bucket_counts(population: Population) -> np.ndarray:
    samples = kept_samples(population)
    counts = np.zeros(K_STUDENT_SAMPLES + 1, dtype=int)
    for s in samples:
        counts[s.get_data()["difficulty"]] += 1
    return counts


def enough_samples(population: Population, num_samples: int) -> bool:
    return len(kept_samples(population)) >= num_samples


# --- driving the loop ---


def run_evolution(population: Population, engine: EvolutionEngine, num_samples: int, max_generations: int) -> int:
    generation = 0
    while not enough_samples(population, num_samples) and generation < max_generations:
        result = engine.step()
        generation += 1
        counts = bucket_counts(population)
        logger.info(
            "gen %d: candidates=%d accepted=%d selected=%d total=%d bucket_counts=%s",
            generation,
            result.num_candidates,
            result.num_accepted,
            result.num_selected,
            len(kept_samples(population)),
            counts.tolist(),
        )
        if generation % CHECKPOINT_EVERY_N_GENERATIONS == 0:
            checkpoint_synthetic_arm(population, generation)
    return generation


def checkpoint_synthetic_arm(population: Population, generation: int) -> None:
    frame = write_synthetic_arm(population)
    logger.info("checkpointed %d synthetic samples after generation %d", len(frame), generation)


def synthetic_arm_dataframe(population: Population) -> pd.DataFrame:
    rows = [s.get_data() for s in kept_samples(population)]
    return pd.DataFrame(rows)


def write_synthetic_arm(population: Population) -> pd.DataFrame:
    frame = synthetic_arm_dataframe(population)
    SYNTHETIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(SYNTHETIC_DATA_PATH)
    return frame


def plot_difficulty_distribution(population: Population) -> None:
    plot_difficulty_histogram(
        bucket_counts(population),
        title="Synthetic dataset difficulty distribution",
        output_path=DIFFICULTY_PLOT_PATH,
    )


# --- main ---


def main(
    smoke_test: bool = False,
    wandb: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    num_samples: int = NUM_SAMPLES,
) -> None:
    num_samples = 6 if smoke_test else num_samples
    num_seeds = 2 if smoke_test else NUM_SEEDS
    num_mutation_samples = 2 if smoke_test else NUM_MUTATION_SAMPLES
    mutations_per_parent = 2 if smoke_test else MUTATIONS_PER_PARENT
    num_crossover_samples = 2 if smoke_test else NUM_CROSSOVER_SAMPLES
    crossover_children_per_pair = 2 if smoke_test else CROSSOVER_CHILDREN_PER_PAIR
    selection_size = 5 if smoke_test else SELECTION_SIZE

    teacher = TeacherClient()
    student = StudentClient(StudentConfig())
    student.load()

    logger.info("Loading and scoring %d seed questions...", num_seeds)
    seeds = load_seeds(teacher, student, n=num_seeds)

    population = Population()
    population.set_seeds(seeds)
    population.set_embedding_column(embedding_fn)

    engine = EvolutionEngine(population, selection_size=selection_size, log_path=str(LOG_DIR))
    engine.set_mutation(make_mut_fn(teacher, student, mutations_per_parent), num_mutation_samples=num_mutation_samples)
    engine.set_crossover(
        make_crossover_fn(teacher, student, crossover_children_per_pair), num_crossover_samples=num_crossover_samples
    )
    engine.set_filters([format_filter_fn])
    engine.set_selection_config(SELECTION_CONFIG)
    if wandb:
        engine.set_wandb_logging(project=wandb_project, config=SELECTION_CONFIG)

    generations = run_evolution(population, engine, num_samples, MAX_GENERATIONS_SAFETY_CAP)
    engine.finish_wandb()
    logger.info(
        "Evolution finished after %d generations. Archive size: %d. Bucket counts: %s",
        generations,
        len(population.get_archive()),
        bucket_counts(population).tolist(),
    )

    synthetic_frame = write_synthetic_arm(population)
    logger.info("Wrote %d synthetic samples to %s", len(synthetic_frame), SYNTHETIC_DATA_PATH)

    plot_difficulty_distribution(population)
    logger.info("Wrote difficulty distribution plot to %s", DIFFICULTY_PLOT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end pass (2 seeds, 3 samples/bucket target) to verify wiring before a full run.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="wandb project name (used when --wandb is set).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Stop once this many samples have been accepted (default: {NUM_SAMPLES}).",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, wandb=args.wandb, wandb_project=args.wandb_project, num_samples=args.num_samples)
