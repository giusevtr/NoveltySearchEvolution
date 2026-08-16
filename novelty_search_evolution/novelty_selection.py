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

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D (n_samples, n_features) array, got shape {X.shape}")
    if X.shape[0] == 0:
        return np.empty((0,))

    if reference is None:
        if X.shape[0] == 1:
            # Nothing to compare against: a lone row has no neighbors once self is excluded.
            raise ValueError(
                "Cannot score novelty for a single row against itself; pass a non-empty "
                "`reference` set"
            )
        dists = cdist(X, X, metric=distance)
        np.fill_diagonal(dists, np.inf)
        pool_size = X.shape[0] - 1
    else:
        if reference.ndim != 2:
            raise ValueError(
                f"reference must be a 2-D (n_samples, n_features) array, got shape "
                f"{reference.shape}"
            )
        if reference.shape[0] == 0:
            raise ValueError("reference is empty; pass reference=None to score X against itself")
        if reference.shape[1] != X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but reference has {reference.shape[1]}; "
                "embeddings must share a single fixed length"
            )
        dists = cdist(X, reference, metric=distance)
        pool_size = reference.shape[0]

    if np.isnan(dists).any():
        # cdist yields NaN for e.g. cosine distance against an all-zero vector; left alone it
        # silently makes every downstream novelty comparison meaningless.
        raise ValueError(
            f"{distance} distances contain NaN — check for zero/degenerate embedding vectors"
        )

    eff_k = min(k, pool_size)
    nearest = np.sort(dists, axis=1)[:, :eff_k]
    return nearest.mean(axis=1)
