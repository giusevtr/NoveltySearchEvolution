"""Embedding-space diversity metrics — domain-agnostic, operate on any (N, D) embedding array
produced by a user's embedding_fn."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

_SUBSAMPLE_THRESHOLD = 1000


def _subsampled_mean(
    embeddings: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    max_samples: int = _SUBSAMPLE_THRESHOLD,
    seed: int = 0,
) -> float:
    """Average `metric_fn` over ceil(N / max_samples) random subsamples of size max_samples,
    instead of running it on all N embeddings — keeps O(N^2)/O(N^3) metrics tractable at scale
    while still drawing from the full population across repeats."""
    n = len(embeddings)
    if n <= max_samples:
        return metric_fn(embeddings)
    rng = np.random.default_rng(seed)
    num_repeats = math.ceil(n / max_samples)
    scores = [
        metric_fn(embeddings[rng.choice(n, size=max_samples, replace=False)])
        for _ in range(num_repeats)
    ]
    return float(np.mean(scores))


def _average_pairwise_distance_full(embeddings: np.ndarray) -> float:
    return float(np.mean(pdist(embeddings, metric="cosine")))


def average_pairwise_distance(embeddings: np.ndarray) -> float:
    """Mean cosine distance over all pairs — higher means more internally diverse."""
    if len(embeddings) < 2:
        raise ValueError(f"need at least 2 embeddings to average pairwise distance, got {len(embeddings)}")
    return _subsampled_mean(embeddings, _average_pairwise_distance_full)


def _vendi_score_full(embeddings: np.ndarray) -> float:
    similarity = 1.0 - squareform(pdist(embeddings, metric="cosine"))
    eigenvalues = np.linalg.eigvalsh(similarity / len(embeddings))
    eigenvalues = eigenvalues[eigenvalues > 1e-12]  # guard log(0) from numerical-noise negatives/zeros
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def vendi_score(embeddings: np.ndarray) -> float:
    """Effective number of distinct samples: exp(entropy) of the eigenvalues of the
    mean-normalized cosine-similarity kernel — a redundancy-aware diversity measure."""
    if len(embeddings) < 2:
        raise ValueError(f"need at least 2 embeddings for Vendi score, got {len(embeddings)}")
    return _subsampled_mean(embeddings, _vendi_score_full)


def _mean_min_distance_to_reference_full(embeddings: np.ndarray, reference_embeddings: np.ndarray) -> float:
    dists = cdist(reference_embeddings, embeddings, metric="cosine")
    return float(dists.min(axis=1).mean())


def mean_min_distance_to_reference(
    embeddings: np.ndarray, reference_embeddings: np.ndarray, max_samples: int = _SUBSAMPLE_THRESHOLD, seed: int = 0
) -> float:
    """For each reference (ground-truth) embedding, the cosine distance to its nearest neighbor
    in `embeddings`, averaged over all reference embeddings — lower means every ground-truth
    point has some synthetic point nearby, i.e. better coverage of the reference distribution."""
    if len(embeddings) == 0 or len(reference_embeddings) == 0:
        raise ValueError("both embeddings and reference_embeddings must be non-empty")
    n, m = len(embeddings), len(reference_embeddings)
    if n <= max_samples and m <= max_samples:
        return _mean_min_distance_to_reference_full(embeddings, reference_embeddings)
    rng = np.random.default_rng(seed)
    num_repeats = math.ceil(max(n, m) / max_samples)
    scores = []
    for _ in range(num_repeats):
        emb_sample = embeddings[rng.choice(n, size=min(n, max_samples), replace=False)]
        ref_sample = reference_embeddings[rng.choice(m, size=min(m, max_samples), replace=False)]
        scores.append(_mean_min_distance_to_reference_full(emb_sample, ref_sample))
    return float(np.mean(scores))
