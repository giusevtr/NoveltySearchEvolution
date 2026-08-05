"""Evaluate the base student, a LoRA-distilled student, or the teacher on GSM8K val.

Draws k samples per question via build_solve_prompt, parses the final '#### <number>' line from
each sample and the ground-truth completion, and reports mean_correctness_rate (fraction of k
samples correct, averaged over questions) and pass_at_k_rate (fraction of questions with at
least one correct sample).

Usage:
    PYTHONPATH=. python examples/gsm8k_distillation/eval.py --mode student --run-name base
    PYTHONPATH=. python examples/gsm8k_distillation/eval.py --mode student \
        --lora-path checkpoints/synthetic --run-name synthetic
    PYTHONPATH=. python examples/gsm8k_distillation/eval.py --mode teacher --run-name teacher

student mode requires a CUDA GPU; teacher mode requires AWS Bedrock credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from examples.gsm8k_distillation.clients.student_client import StudentClient, StudentConfig
from examples.gsm8k_distillation.clients.teacher_client import TeacherClient
from examples.gsm8k_distillation.novelty_search.prompts import answers_match, build_solve_prompt, extract_gsm8k_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
DEFAULT_DATA_PATH = THIS_DIR / "data" / "gsm8k_val.parquet"

_STUDENT_TEMPERATURE = 0.8
_TEACHER_TEMPERATURE = 0.7
_MAX_TOKENS = 512


class ParsingErrorCounter:
    def __init__(self) -> None:
        self.num_samples = 0
        self.num_parsing_errors = 0

    def record(self, predicted: str | None) -> None:
        self.num_samples += 1
        if predicted is None:
            self.num_parsing_errors += 1


def generate_student(questions: list[str], k: int, lora_path: str | None) -> list[list[str]]:
    student = StudentClient(StudentConfig(lora_path=lora_path))
    student.load()
    prompts = [build_solve_prompt(q) for q in questions]
    return student.generate(prompts, n=k, temperature=_STUDENT_TEMPERATURE, max_tokens=_MAX_TOKENS)


def generate_teacher(questions: list[str], k: int) -> list[list[str]]:
    teacher = TeacherClient()
    prompts = [build_solve_prompt(q) for q in questions]
    per_repeat = [teacher.generate(prompts, max_tokens=_MAX_TOKENS, temperature=_TEACHER_TEMPERATURE) for _ in range(k)]
    return [[per_repeat[r][i] for r in range(k)] for i in range(len(questions))]


def evaluate(
    df: pd.DataFrame, k: int, mode: str, lora_path: str | None, error_counter: ParsingErrorCounter
) -> pd.DataFrame:
    questions = df["prompt"].tolist()
    ground_truth = df["completion"].tolist()

    all_samples = generate_student(questions, k, lora_path) if mode == "student" else generate_teacher(questions, k)

    rows = []
    for samples, gt in zip(all_samples, ground_truth):
        true_answer = extract_gsm8k_answer(gt)
        correctness_flags = []
        for s in samples:
            predicted = extract_gsm8k_answer(s)
            error_counter.record(predicted)
            correctness_flags.append(answers_match(predicted, true_answer))
        rows.append({"correctness_rate": sum(correctness_flags) / k, "pass_at_k": int(any(correctness_flags))})
    return pd.DataFrame(rows)[["correctness_rate", "pass_at_k"]]


def build_report(
    annotated: pd.DataFrame, k: int, error_counter: ParsingErrorCounter, mode: str, model: str, lora_path: str | None, run_name: str
) -> dict:
    num_questions = len(annotated)
    num_samples = error_counter.num_samples
    return {
        "run_name": run_name,
        "mode": mode,
        "model": model,
        "lora_path": lora_path,
        "num_questions": num_questions,
        "k": k,
        "mean_correctness_rate": float(annotated["correctness_rate"].mean()) if num_questions else 0.0,
        "pass_at_k_rate": float(annotated["pass_at_k"].mean()) if num_questions else 0.0,
        "num_parsing_errors": error_counter.num_parsing_errors,
        "parsing_error_rate": error_counter.num_parsing_errors / num_samples if num_samples else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["student", "teacher"], required=True)
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA adapter checkpoint (student mode only).")
    parser.add_argument("--k", type=int, default=1, help="Samples drawn per question.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N val questions.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--run-name", type=str, required=True, help="Namespaces output under output/eval/<run-name>/.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Evaluate only a handful of questions.")
    args = parser.parse_args()

    if args.mode == "teacher" and args.lora_path is not None:
        parser.error("--lora-path is not valid with --mode teacher")

    df = pd.read_parquet(args.data)
    limit = args.limit
    if args.smoke_test:
        limit = min(4, limit) if limit is not None else 4
    if limit is not None:
        df = df.head(limit)

    logger.info("Evaluating %s (mode=%s, lora_path=%s) on %d questions, k=%d", args.run_name, args.mode, args.lora_path, len(df), args.k)

    error_counter = ParsingErrorCounter()
    annotated = evaluate(df, args.k, args.mode, args.lora_path, error_counter)

    output_dir = args.output_dir or (THIS_DIR / "output" / "eval" / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_json(output_dir / "annotated.jsonl", orient="records", lines=True)

    model = StudentConfig().name if args.mode == "student" else "teacher (Bedrock)"
    report = build_report(annotated, args.k, error_counter, args.mode, model, args.lora_path, args.run_name)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    logger.info(
        "mean_correctness_rate=%.3f pass_at_%d_rate=%.3f parsing_error_rate=%.3f",
        report["mean_correctness_rate"], args.k, report["pass_at_k_rate"], report["parsing_error_rate"],
    )
    logger.info("Wrote report to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
