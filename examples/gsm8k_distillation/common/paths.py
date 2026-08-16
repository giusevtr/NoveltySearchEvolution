"""Filesystem layout shared by every script in this example."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "output"
LOG_DIR = PACKAGE_DIR / "logs"
CONFIG_DIR = PACKAGE_DIR / "configs"

GSM8K_TRAIN_PARQUET = DATA_DIR / "gsm8k_train.parquet"
GSM8K_VAL_PARQUET = DATA_DIR / "gsm8k_val.parquet"
