"""Stateless novelty scoring primitive: mean distance to k nearest neighbors."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.spatial.distance import cdist

_METRICS = {"cosine", "euclidean"}


def score(
    X: np.ndarray,
    k: int,
    distance: Literal["cosine", "euclidean"] = "cosine",
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a novelty score per row of X: mean distance to its k nearest neighbors in
    `reference` (or in X itself, excluding self, if reference is None). Higher == more novel.
    """
    if distance not in _METRICS:
        raise ValueError(f"Unknown distance metric: {distance!r}. Must be one of {_METRICS}")

    if reference is None:
        dists = cdist(X, X, metric=distance)
        np.fill_diagonal(dists, np.inf)
        pool_size = X.shape[0] - 1
    else:
        dists = cdist(X, reference, metric=distance)
        pool_size = reference.shape[0]

    eff_k = max(1, min(k, pool_size))
    nearest = np.sort(dists, axis=1)[:, :eff_k]
    return nearest.mean(axis=1)
