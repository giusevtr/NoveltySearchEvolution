"""GSM8K distillation data augmentation via novelty search.

Mutates 5 seed GSM8K questions using a Bedrock-hosted Qwen3-32B teacher, generates teacher
completions (ground truth), scores each question's difficulty by sampling K=10 completions
from a colocated vLLM Qwen3-0.6B student and counting how many match the teacher's answer,
embeds each sample as a concatenation of unit-norm question semantics (Bedrock Titan text
embeddings) and a unit-norm one-hot difficulty encoding, filters out samples with malformed
teacher completions, and
grows the population via `novelty_search_evolution`'s novelty search engine until every
difficulty bucket (0..K) has at least MIN_SAMPLES_PER_BUCKET accepted samples.

Produces `output/synthetic_distillation.parquet` and a difficulty-distribution plot.

Usage:
    PYTHONPATH=. python examples/gsm8k_distillation/run.py [--smoke-test]

Requires AWS Bedrock credentials (standard boto3 chain) for the Qwen3-32B teacher, and a
CUDA GPU for the colocated Qwen3-0.6B vLLM student.
"""

from __future__ import annotations
import sys
import argparse
import logging
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.gsm8k_distillation.clients.student_client import StudentClient, StudentConfig
from examples.gsm8k_distillation.clients.teacher_client import TeacherClient
from examples.gsm8k_distillation.novelty_search.embedding import combine_embeddings, question_embedding_fn
from examples.gsm8k_distillation.novelty_search.prompts import (
    TOPICS,
    answers_match,
    build_crossover_prompt,
    build_mutation_prompt,
    build_solve_prompt,
    extract_gsm8k_answer,
    extract_question,
)

from novelty_search_evolution import EvolutionEngine, EvoSample, Population

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- config ---

NUM_SEEDS = 5
K_STUDENT_SAMPLES = 10  # difficulty label range: [0, K_STUDENT_SAMPLES]
MIN_SAMPLES_PER_BUCKET = 50
MAX_GENERATIONS_SAFETY_CAP = 300

MUTATIONS_PER_PARENT = 5
NUM_MUTATION_SAMPLES = 10  # parents sampled per generation
CROSSOVER_CHILDREN_PER_PAIR = 5
NUM_CROSSOVER_SAMPLES = 5  # parent pairs sampled per generation
SELECTION_SIZE = 15

SELECTION_CONFIG = {
    "nn-k": MIN_SAMPLES_PER_BUCKET,
    "distance": "cosine",
    "archive_update_prob": 0.3,
}

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
LOG_DIR = Path(__file__).parent / "logs"
CHECKPOINT_EVERY_N_GENERATIONS = 10

_MUTATION_TEMPERATURE = 0.9
_TEACHER_SOLVE_TEMPERATURE = 0.7
_STUDENT_SOLVE_TEMPERATURE = 0.8
_MUTATION_MAX_TOKENS = 256
_SOLVE_MAX_TOKENS = 512


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


def score_difficulty(student: StudentClient, questions: list[str], teacher_answers: list[str | None]) -> list[int]:
    """For each question, sample K student completions and count matches vs the teacher answer."""
    prompts = [build_solve_prompt(q) for q in questions]
    student_completions = student.generate(
        prompts, n=K_STUDENT_SAMPLES, temperature=_STUDENT_SOLVE_TEMPERATURE, max_tokens=_SOLVE_MAX_TOKENS
    )
    difficulties = []
    for completions, teacher_answer in zip(student_completions, teacher_answers):
        matches = sum(1 for c in completions if answers_match(extract_gsm8k_answer(c), teacher_answer))
        difficulties.append(matches)
    return difficulties


def load_seeds(teacher: TeacherClient, student: StudentClient, n: int, random_state: int = 0) -> list[dict]:
    df = pd.read_parquet(DATA_DIR / "gsm8k_train.parquet")
    seed_questions = df["prompt"].sample(n=n, random_state=random_state).tolist()

    teacher_completions = teacher.generate(
        [build_solve_prompt(q) for q in seed_questions], max_tokens=_SOLVE_MAX_TOKENS, temperature=_TEACHER_SOLVE_TEMPERATURE
    )
    teacher_answers = [extract_gsm8k_answer(c) for c in teacher_completions]
    difficulties = score_difficulty(student, seed_questions, teacher_answers)

    return [
        _assemble_dict(q, c, d, source="seed", parent_question=None, topic=None)
        for q, c, d in zip(seed_questions, teacher_completions, difficulties)
    ]


# --- evolution operators ---


def make_mut_fn(teacher: TeacherClient, student: StudentClient, mutations_per_parent: int):
    def mut_fn(parent: EvoSample) -> list[dict]:
        parent_q = parent.get_data()["question"]
        topics = [random.choice(TOPICS) for _ in range(mutations_per_parent)]
        mutation_prompts = [build_mutation_prompt(parent_q, topic) for topic in topics]
        raw_questions = teacher.generate(mutation_prompts, max_tokens=_MUTATION_MAX_TOKENS, temperature=_MUTATION_TEMPERATURE)
        new_questions = [extract_question(q) for q in raw_questions]

        teacher_completions = teacher.generate(
            [build_solve_prompt(q) for q in new_questions], max_tokens=_SOLVE_MAX_TOKENS, temperature=_TEACHER_SOLVE_TEMPERATURE
        )
        teacher_answers = [extract_gsm8k_answer(c) for c in teacher_completions]
        difficulties = score_difficulty(student, new_questions, teacher_answers)

        return [
            _assemble_dict(q, c, d, source="mutation", parent_question=parent_q, topic=topic)
            for q, c, d, topic in zip(new_questions, teacher_completions, difficulties, topics)
        ]

    return mut_fn


def make_crossover_fn(teacher: TeacherClient, student: StudentClient, children_per_pair: int):
    def crossover_fn(parent_1: EvoSample, parent_2: EvoSample) -> list[dict]:
        q1 = parent_1.get_data()["question"]
        q2 = parent_2.get_data()["question"]
        topics = [random.choice(TOPICS) for _ in range(children_per_pair)]
        crossover_prompts = [build_crossover_prompt(q1, q2, topic) for topic in topics]
        raw_questions = teacher.generate(crossover_prompts, max_tokens=_MUTATION_MAX_TOKENS, temperature=_MUTATION_TEMPERATURE)
        new_questions = [extract_question(q) for q in raw_questions]

        teacher_completions = teacher.generate(
            [build_solve_prompt(q) for q in new_questions], max_tokens=_SOLVE_MAX_TOKENS, temperature=_TEACHER_SOLVE_TEMPERATURE
        )
        teacher_answers = [extract_gsm8k_answer(c) for c in teacher_completions]
        difficulties = score_difficulty(student, new_questions, teacher_answers)

        return [
            _assemble_dict(q, c, d, source="crossover", parent_question=q1, parent_question_2=q2, topic=topic)
            for q, c, d, topic in zip(new_questions, teacher_completions, difficulties, topics)
        ]

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
    """Archived ∪ active samples, de-duplicated: a sample can be both at once."""
    by_id = {s.get_id(): s for s in population.get_archive() + population.get_active()}
    return list(by_id.values())


def bucket_counts(population: Population) -> np.ndarray:
    samples = kept_samples(population)
    counts = np.zeros(K_STUDENT_SAMPLES + 1, dtype=int)
    for s in samples:
        counts[s.get_data()["difficulty"]] += 1
    return counts


def buckets_satisfied(population: Population, min_per_bucket: int) -> bool:
    return bool(np.all(bucket_counts(population) >= min_per_bucket))


# --- driving the loop ---


def run_evolution(population: Population, engine: EvolutionEngine, min_per_bucket: int, max_generations: int) -> int:
    generation = 0
    while not buckets_satisfied(population, min_per_bucket) and generation < max_generations:
        result = engine.step()
        generation += 1
        counts = bucket_counts(population)
        logger.info(
            "gen %d: candidates=%d accepted=%d selected=%d bucket_counts=%s",
            generation,
            result.num_candidates,
            result.num_accepted,
            result.num_selected,
            counts.tolist(),
        )
        if generation % CHECKPOINT_EVERY_N_GENERATIONS == 0:
            checkpoint_synthetic_arm(population, generation)
    return generation


def checkpoint_synthetic_arm(population: Population, generation: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = synthetic_arm_dataframe(population)
    frame.to_parquet(OUTPUT_DIR / "synthetic_distillation.parquet")
    logger.info("checkpointed %d synthetic samples after generation %d", len(frame), generation)


def synthetic_arm_dataframe(population: Population) -> pd.DataFrame:
    rows = [s.get_data() for s in kept_samples(population)]
    return pd.DataFrame(rows)


def plot_difficulty_distribution(population: Population) -> None:
    counts = bucket_counts(population)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(K_STUDENT_SAMPLES + 1), counts)
    ax.axhline(MIN_SAMPLES_PER_BUCKET, color="red", linestyle="--", linewidth=1, label=f"target ({MIN_SAMPLES_PER_BUCKET})")
    ax.set_xlabel("difficulty (# of K student successes)")
    ax.set_ylabel("sample count")
    ax.set_title("Synthetic dataset difficulty distribution")
    ax.legend()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "difficulty_distribution.png", dpi=120)
    plt.close(fig)


# --- main ---


def main(smoke_test: bool = False) -> None:
    min_per_bucket = 3 if smoke_test else MIN_SAMPLES_PER_BUCKET
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

    generations = run_evolution(population, engine, min_per_bucket, MAX_GENERATIONS_SAFETY_CAP)
    logger.info(
        "Evolution finished after %d generations. Archive size: %d. Bucket counts: %s",
        generations,
        len(population.get_archive()),
        bucket_counts(population).tolist(),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_frame = synthetic_arm_dataframe(population)
    synthetic_frame.to_parquet(OUTPUT_DIR / "synthetic_distillation.parquet")
    logger.info("Wrote %d synthetic samples to %s", len(synthetic_frame), OUTPUT_DIR / "synthetic_distillation.parquet")

    plot_difficulty_distribution(population)
    logger.info("Wrote difficulty distribution plot to %s", OUTPUT_DIR / "difficulty_distribution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end pass (2 seeds, 3 samples/bucket target) to verify wiring before a full run.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
