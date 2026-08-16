"""JSON/JSONL/parquet helpers shared by the example's scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

# Plain getLogger, not logging_utils.get_logger: only entrypoint scripts should configure logging.
logger = logging.getLogger(__name__)


def read_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text())


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_frame(path: Path, n: int | None = None, random_state: int = 0) -> pd.DataFrame:
    """Read a parquet/JSONL dataset, optionally down-sampled to `n` rows with a fixed seed."""
    frame = pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_json(path, lines=True)
    if n is not None and n < len(frame):
        frame = frame.sample(n=n, random_state=random_state)
    return frame.reset_index(drop=True)


def load_seed_questions(path: Path, n: int, random_state: int = 0, column: str = "prompt") -> list[str]:
    return load_frame(path, n=n, random_state=random_state)[column].tolist()


def write_frame(frame: pd.DataFrame, directory: Path, basename: str) -> tuple[Path, Path]:
    """Write `frame` as both `<basename>.parquet` and `<basename>.jsonl`, returning both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    parquet_path = directory / f"{basename}.parquet"
    jsonl_path = directory / f"{basename}.jsonl"
    frame.to_parquet(parquet_path)
    frame.to_json(jsonl_path, orient="records", lines=True)
    return parquet_path, jsonl_path


def cap_limit(limit: int | None, cap: int) -> int:
    """Tighten an optional row limit to at most `cap` (used by every --smoke-test flag)."""
    return min(cap, limit) if limit is not None else cap


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        logger.error("Missing %s — %s", path, hint)
        raise SystemExit(1)
