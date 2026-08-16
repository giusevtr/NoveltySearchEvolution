"""Stateless novelty scoring primitive: mean distance to k nearest neighbors."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.spatial.distance import cdist

_METRICS = {"cosine", "euclidean"}


def _reject_zero_rows(X: np.ndarray, name: str) -> None:
    """Cosine distance is undefined for zero-norm vectors (scipy returns NaN)."""
    if X.size and np.any(np.linalg.norm(X, axis=1) == 0):
        zero_rows = np.flatnonzero(np.linalg.norm(X, axis=1) == 0).tolist()
        raise ValueError(
            f"{name} contains zero-norm row(s) {zero_rows}, which cosine distance cannot "
            "score (they would produce NaN distances). Drop them or use distance='euclidean'."
        )


def score(
    X: np.ndarray,
    k: int,
    distance: Literal["cosine", "euclidean"] = "cosine",
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a novelty score per row of X: mean distance to its k nearest neighbors in
    `reference` (or in X itself, excluding self, if reference is None). Higher == more novel.

    When there is nothing to compare against (a single row and no reference, or an empty
    reference) every row scores 0.0: no row is more novel than any other.
    """
    if distance not in _METRICS:
        raise ValueError(f"Unknown distance metric: {distance!r}. Must be one of {_METRICS}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    if distance == "cosine":
        _reject_zero_rows(X, "X")
        if reference is not None:
            _reject_zero_rows(reference, "reference")

    if reference is None:
        dists = cdist(X, X, metric=distance)
        np.fill_diagonal(dists, np.inf)
        pool_size = X.shape[0] - 1
    else:
        if reference.ndim != 2:
            raise ValueError(f"reference must be 2-dimensional, got shape {reference.shape}")
        dists = cdist(X, reference, metric=distance)
        pool_size = reference.shape[0]

    if pool_size == 0:
        return np.zeros(X.shape[0], dtype=np.float64)

    eff_k = max(1, min(k, pool_size))
    nearest = np.sort(dists, axis=1)[:, :eff_k]
    return nearest.mean(axis=1)
