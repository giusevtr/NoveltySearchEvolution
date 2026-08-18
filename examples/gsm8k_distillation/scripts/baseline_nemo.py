"""Baseline synthetic-data generator using NVIDIA NeMo Data Designer.

Non-novelty-search comparison arm: generates flat, independent grade-school math questions
conditioned on the same 5 seed GSM8K questions used by run_novelty_search_augmentatin.py (embedded as static few-shot
examples) plus Data Designer's native diversity mechanism (SamplerColumnConfig columns for topic
and step-count, sampled per record and referenced in the generation prompt via Jinja) — rather
than novelty search's iterative single-parent mutation + selection loop. Generation and solving
both go through the same Bedrock-hosted Qwen3-32B teacher as run_novelty_search_augmentatin.py (routed through a local
OpenAI-compatible proxy, clients/teacher_openai_proxy.py, so Data Designer can call it), and the
teacher_completion prompt/inference params are imported directly from run_novelty_search_augmentatin.py so they can't drift
out of sync with novelty search's teacher-solve step.

Produces output/baseline_nemo_distillation.{parquet,jsonl}, schema-compatible with
output/synthetic_distillation.parquet (question/teacher_completion/final_answer), trainable via:
    python examples/gsm8k_distillation/train.py \
        --data output/baseline_nemo_distillation.parquet \
        --prompt-column question --completion-column teacher_completion \
        --output-dir checkpoints/baseline_nemo

Usage:
    PYTHONPATH=. python examples/gsm8k_distillation/baseline_nemo.py [--smoke-test] [--num-samples N]

Requires AWS Bedrock credentials (standard boto3 chain) and `pip install data-designer fastapi
uvicorn`.
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time

import data_designer.config as dd
import numpy as np
import pandas as pd
from data_designer.interface import DataDesigner

from examples.gsm8k_distillation.clients.student_client import StudentClient, StudentConfig
from examples.gsm8k_distillation.clients.teacher_openai_proxy import serve as serve_proxy
from examples.gsm8k_distillation.common.files import load_seed_questions, write_frame
from examples.gsm8k_distillation.common.inference import (
    K_STUDENT_SAMPLES,
    MUTATION_MAX_TOKENS,
    MUTATION_TEMPERATURE,
    SOLVE_MAX_TOKENS,
    TEACHER_SOLVE_TEMPERATURE,
    score_difficulty,
)
from examples.gsm8k_distillation.common.logging_utils import get_logger
from examples.gsm8k_distillation.common.paths import GSM8K_TRAIN_PARQUET, OUTPUT_DIR
from examples.gsm8k_distillation.common.plots import plot_difficulty_histogram
from examples.gsm8k_distillation.novelty_search.prompts import (
    TOPICS,
    build_baseline_generation_prompt,
    build_solve_prompt,
    extract_gsm8k_answer,
)
from examples.gsm8k_distillation.scripts.run_novelty_search_augmentatin import NUM_SEEDS

logger = get_logger(__name__)

DEFAULT_NUM_SAMPLES = 8000
SMOKE_TEST_NUM_RECORDS = 4
OUTPUT_BASENAME = "baseline_nemo_distillation"
DIFFICULTY_PLOT_PATH = OUTPUT_DIR / "nemo_difficulty_distribution.png"

_PROXY_API_KEY_ENV = "TEACHER_PROXY_DUMMY_KEY"

_DIFFICULTIES = ["very easy", "easy", "medium", "hard", "very hard"]
# num_steps is sampled conditionally on difficulty via SubcategorySamplerParams: harder problems
# are given more reasoning steps to solve, with adjacent difficulty levels overlapping in step
# count so the mapping isn't a rigid 1:1 rule (difficulty depends on more than step count alone).
_NUM_STEPS_BY_DIFFICULTY = {
    "very easy": ["one step"],
    "easy": ["one step", "two steps"],
    "medium": ["two steps", "three steps"],
    "hard": ["three steps", "several steps"],
    "very hard": ["several steps"],
}


@dd.custom_column_generator(required_columns=["teacher_completion"])
def _extract_final_answer(row) -> dict:
    return {**row, "final_answer": extract_gsm8k_answer(row["teacher_completion"])}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_proxy() -> int:
    port = _free_port()
    thread = threading.Thread(target=serve_proxy, args=(port,), daemon=True)
    thread.start()
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return port
        time.sleep(0.1)
    raise RuntimeError("Teacher proxy did not come up in time")


def generate_dataset(seed_questions: list[str], num_records: int, proxy_port: int) -> pd.DataFrame:
    os.environ[_PROXY_API_KEY_ENV] = "unused"
    provider = dd.ModelProvider(
        name="teacher-proxy",
        endpoint=f"http://127.0.0.1:{proxy_port}/v1",
        provider_type="openai",
        api_key=_PROXY_API_KEY_ENV,
    )
    generate_cfg = dd.ModelConfig(
        alias="teacher-generate",
        model="qwen.qwen3-32b-v1:0",
        provider="teacher-proxy",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=MUTATION_TEMPERATURE, max_tokens=MUTATION_MAX_TOKENS
        ),
    )
    solve_cfg = dd.ModelConfig(
        alias="teacher-solve",
        model="qwen.qwen3-32b-v1:0",
        provider="teacher-proxy",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=TEACHER_SOLVE_TEMPERATURE, max_tokens=SOLVE_MAX_TOKENS
        ),
    )

    config_builder = dd.DataDesignerConfigBuilder(model_configs=[generate_cfg, solve_cfg])
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=TOPICS)
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="difficulty", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=_DIFFICULTIES)
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="num_steps",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(category="difficulty", values=_NUM_STEPS_BY_DIFFICULTY),
        )
    )
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="question",
            prompt=build_baseline_generation_prompt(
                seed_questions, "{{ topic }}", "{{ difficulty }}", "{{ num_steps }}"
            ),
            model_alias="teacher-generate",
        )
    )
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="teacher_completion",
            prompt=build_solve_prompt("{{ question }}"),
            model_alias="teacher-solve",
        )
    )
    config_builder.add_column(
        dd.CustomColumnConfig(name="final_answer", generator_function=_extract_final_answer)
    )

    data_designer = DataDesigner(model_providers=[provider])
    result = data_designer.create(
        config_builder=config_builder, num_records=num_records, dataset_name="gsm8k_baseline_questions"
    )
    return result.load_dataset()


def score_student_difficulty(frame: pd.DataFrame) -> pd.DataFrame:
    """Score each question's difficulty by sampling the student K times, matching run_novelty_search_augmentatin.py exactly."""
    student = StudentClient(StudentConfig())
    student.load()
    difficulties = score_difficulty(student, frame["question"].tolist(), frame["final_answer"].tolist())
    frame = frame.copy()
    frame["student_difficulty"] = difficulties
    return frame


def plot_nemo_difficulty_distribution(frame: pd.DataFrame) -> None:
    plot_difficulty_histogram(
        np.bincount(frame["student_difficulty"], minlength=K_STUDENT_SAMPLES + 1),
        title="NeMo baseline dataset difficulty distribution",
        output_path=DIFFICULTY_PLOT_PATH,
    )


def main(smoke_test: bool = False, num_records: int | None = None) -> None:
    num_seeds = 2 if smoke_test else NUM_SEEDS
    if num_records is None:
        num_records = SMOKE_TEST_NUM_RECORDS if smoke_test else DEFAULT_NUM_SAMPLES

    seed_questions = load_seed_questions(GSM8K_TRAIN_PARQUET, n=num_seeds)
    logger.info("Loaded %d seed questions", len(seed_questions))

    proxy_port = _start_proxy()
    logger.info("Teacher proxy listening on 127.0.0.1:%d", proxy_port)

    frame = generate_dataset(seed_questions, num_records=num_records, proxy_port=proxy_port)
    logger.info("Data Designer generated %d rows", len(frame))

    frame = frame.copy()
    frame["source"] = "nemo_baseline"
    frame = frame[frame["final_answer"].notna()].reset_index(drop=True)
    frame = score_student_difficulty(frame)
    logger.info("Kept %d of %d generated questions after filtering malformed completions", len(frame), num_records)

    parquet_path, jsonl_path = write_frame(frame, OUTPUT_DIR, OUTPUT_BASENAME)
    logger.info("Wrote %d difficulty-scored baseline samples to %s and %s", len(frame), parquet_path, jsonl_path)

    plot_nemo_difficulty_distribution(frame)
    logger.info("Wrote difficulty distribution plot to %s", DIFFICULTY_PLOT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end pass (2 seeds, 4 generated questions) to verify wiring before a full run.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        dest="num_records",
        help=f"Number of questions to generate (default: {DEFAULT_NUM_SAMPLES}, or {SMOKE_TEST_NUM_RECORDS} in --smoke-test).",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, num_records=args.num_records)
