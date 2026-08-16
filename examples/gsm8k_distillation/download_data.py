"""Download GSM8K and write it out as prompt/completion JSONL files.

Downloads `openai/gsm8k` (`main` config) via Hugging Face `datasets`, maps each example to
`{"prompt": question, "completion": answer}` (same transform used in `sft_training.ipynb`),
and writes one JSON object per line to `data/gsm8k_train.jsonl` and `data/gsm8k_val.jsonl`,
alongside this script — the directory every other script in this example reads from.

Usage:
    ./run.sh download_data.py [--limit N]
    # or, from this directory with deps installed:
    python download_data.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"


def to_prompt_completion(example: dict) -> dict:
    return {"prompt": example["question"], "completion": example["answer"]}


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def main(output_dir: Path, limit: int | None) -> None:
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    val_dataset = load_dataset("openai/gsm8k", "main", split="test")

    if limit is not None:
        train_dataset = train_dataset.select(range(min(limit, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(limit, len(val_dataset))))

    train_examples = [to_prompt_completion(example) for example in train_dataset]
    val_examples = [to_prompt_completion(example) for example in val_dataset]

    train_path = output_dir / "gsm8k_train.jsonl"
    val_path = output_dir / "gsm8k_val.jsonl"
    write_jsonl(train_examples, train_path)
    write_jsonl(val_examples, val_path)

    logger.info("Wrote %d train rows to %s", len(train_examples), train_path)
    logger.info("Wrote %d val rows to %s", len(val_examples), val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write the JSONL files."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only keep the first N rows of each split (for a smoke test)."
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, limit=args.limit)
