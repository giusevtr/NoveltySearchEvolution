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
from examples.gsm8k_distillation.common.files import cap_limit, load_frame, write_frame
from examples.gsm8k_distillation.common.inference import teacher_solve
from examples.gsm8k_distillation.common.paths import DATA_DIR
from examples.gsm8k_distillation.novelty_search.prompts import extract_gsm8k_answer

DEFAULT_DATA_PATH = DATA_DIR / "gsm8k_train.jsonl"
DEFAULT_OUTPUT_DIR = DATA_DIR
OUTPUT_BASENAME = "gsm8k_train_teacher_annotated"
SMOKE_TEST_NUM_ROWS = 4

_MAX_TOKENS = 1024


def annotate(questions: list[str], batch_size: int) -> list[str]:
    teacher = TeacherClient()
    completions: list[str] = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        print(f"=== Annotating {start}-{start + len(batch)}/{len(questions)} ===")
        completions.extend(teacher_solve(teacher, batch, max_tokens=_MAX_TOKENS))
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

    limit = cap_limit(args.limit, SMOKE_TEST_NUM_ROWS) if args.smoke_test else args.limit

    df = load_frame(args.data, n=limit, random_state=args.seed)
    completions = annotate(df["prompt"].tolist(), args.batch_size)

    out = pd.DataFrame(
        {
            "question": df["prompt"],
            "teacher_completion": completions,
            "final_answer": [extract_gsm8k_answer(c) for c in completions],
        }
    )

    parquet_path, jsonl_path = write_frame(out, args.output_dir, OUTPUT_BASENAME)

    print(f"=== Done. Annotated {len(out)} rows. Wrote {jsonl_path} and {parquet_path} ===")


if __name__ == "__main__":
    main()
