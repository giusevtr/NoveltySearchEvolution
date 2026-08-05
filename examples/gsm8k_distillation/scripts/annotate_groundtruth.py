"""Annotate ground-truth GSM8K training questions with teacher-model completions.

Mirrors run.py's synthetic-arm schema (question/teacher_completion/final_answer) so the
ground-truth arm can be trained with the same completion style as the synthetic arm, isolating
the question-generation process (novelty search vs. real questions) as the sole difference
between the two arms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd

from examples.gsm8k_distillation.clients.teacher_client import TeacherClient
from examples.gsm8k_distillation.novelty_search.prompts import build_solve_prompt, extract_gsm8k_answer

THIS_DIR = Path(__file__).parent
DEFAULT_DATA_PATH = THIS_DIR.parent / "data" / "gsm8k_train.jsonl"
DEFAULT_OUTPUT_DIR = THIS_DIR.parent / "data"
OUTPUT_BASENAME = "gsm8k_train_teacher_annotated"

_MAX_TOKENS = 1024
_TEACHER_TEMPERATURE = 0.7


def annotate(prompts: list[str], batch_size: int) -> list[str]:
    teacher = TeacherClient()
    completions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        print(f"=== Annotating {start}-{start + len(batch)}/{len(prompts)} ===")
        completions.extend(teacher.generate(batch, max_tokens=_MAX_TOKENS, temperature=_TEACHER_TEMPERATURE))
    return completions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only annotate a random sample of this many rows.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Annotate just a handful of rows.")
    args = parser.parse_args()

    limit = args.limit
    if args.smoke_test:
        limit = min(4, limit) if limit is not None else 4

    df = pd.read_json(args.data, lines=True)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=args.seed).reset_index(drop=True)

    prompts = [build_solve_prompt(q) for q in df["prompt"]]
    completions = annotate(prompts, args.batch_size)

    out = pd.DataFrame(
        {
            "question": df["prompt"],
            "teacher_completion": completions,
            "final_answer": [extract_gsm8k_answer(c) for c in completions],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{OUTPUT_BASENAME}.jsonl"
    parquet_path = args.output_dir / f"{OUTPUT_BASENAME}.parquet"
    out.to_json(jsonl_path, orient="records", lines=True)
    out.to_parquet(parquet_path)

    print(f"=== Done. Annotated {len(out)} rows. Wrote {jsonl_path} and {parquet_path} ===")


if __name__ == "__main__":
    main()
