"""Pluggable selection engine that decides which candidates become `active` each step."""

from __future__ import annotations

from typing import List, Literal, Protocol

import numpy as np

from . import novelty_selection
from .population import Population
from .sample import EvoSample


class SelectionEngine(Protocol):
    def select(self, candidates: List[EvoSample], top_k: int) -> List[EvoSample]:
        """Return up to top_k of `candidates`, chosen by whatever novelty/quality criterion
        the engine implements.
        """
        ...


class DefaultSelectionEngine:
    """Archive-based novelty selection: greedy, sequential picks scored against the archive
    plus every pick already made this round.
    """

    def __init__(
        self,
        population: Population,
        nn_k: int,
        distance: Literal["cosine", "euclidean"] = "cosine",
    ) -> None:
        self.population = population
        self.nn_k = nn_k
        self.distance = distance

    def select(self, candidates: List[EvoSample], top_k: int) -> List[EvoSample]:
        if top_k <= 0 or not candidates:
            return []

        archive_X = self.population.get_archive_embeddings().copy()
        remaining = list(candidates)
        selected: List[EvoSample] = []

        for _ in range(min(top_k, len(remaining))):
            X = self.population.get_embeddings(remaining)
            reference = archive_X if archive_X.size else None
            scores = novelty_selection.score(
                X, k=self.nn_k, distance=self.distance, reference=reference
            )
            best_idx = int(scores.argmax())
            best = remaining.pop(best_idx)
            selected.append(best)
            embedding = best.get_embedding().reshape(1, -1)
            archive_X = (
                np.vstack([archive_X, embedding]) if archive_X.size else embedding.copy()
            )

        return selected
