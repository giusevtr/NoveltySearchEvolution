"""Uniform logging setup for the example's entrypoint scripts."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger(name)
