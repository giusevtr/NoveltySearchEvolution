"""Population — owns every EvoSample ever created and provides status-aware access."""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .sample import EvoSample, Status


class Population:
    def __init__(self) -> None:
        self._by_id: Dict[str, EvoSample] = OrderedDict()
        self._archive: "OrderedDict[str, None]" = OrderedDict()
        self._embedding_fn: Optional[Callable[[List[Any]], List[np.ndarray]]] = None

    # --- setup ---

    def set_seeds(self, seeds: List[Any]) -> None:
        for raw in seeds:
            seed = EvoSample(data=raw, parents=[], generation=0)
            self.add(seed)
            seed.set_active()
            self.add_to_archive(seed)

    def set_embedding_column(self, embedding_fn: Callable[[List[Any]], List[np.ndarray]]) -> None:
        self._embedding_fn = embedding_fn

    # --- mutation ---

    def add(self, sample: EvoSample) -> None:
        self._by_id[sample.get_id()] = sample

    def bulk_add(self, samples: List[EvoSample]) -> None:
        for sample in samples:
            self.add(sample)

    # --- archive ---

    def get_archive(self) -> List[EvoSample]:
        return [self._by_id[sid] for sid in self._archive]

    def add_to_archive(self, sample: EvoSample) -> None:
        self._archive[sample.get_id()] = None

    def get_archive_embeddings(self) -> np.ndarray:
        return self.get_embeddings(self.get_archive())

    # --- read access ---

    def get_all_samples(self) -> List[EvoSample]:
        return list(self._by_id.values())

    def get_active(self) -> List[EvoSample]:
        return self.get_by_status(Status.ACTIVE)

    def get_inactive(self) -> List[EvoSample]:
        return self.get_by_status(Status.INACTIVE)

    def get_stale(self) -> List[EvoSample]:
        return self.get_by_status(Status.STALE)

    def get_rejected(self) -> List[EvoSample]:
        return self.get_by_status(Status.REJECTED)

    def get_by_status(self, status: Status) -> List[EvoSample]:
        return [s for s in self._by_id.values() if s.get_status() == status]

    def get_by_id(self, id: str) -> EvoSample:
        return self._by_id[id]

    def sample_random(self, k: int, status: Optional[Status] = None) -> List[EvoSample]:
        pool = self.get_by_status(status) if status is not None else self.get_all_samples()
        # if k >= len(pool):
        #     return list(pool)
        return random.choices(pool, k=k)

    # --- embeddings ---

    def compute_embeddings(self, samples: Optional[List[EvoSample]] = None) -> None:
        pool = samples if samples is not None else self.get_all_samples()
        missing = [s for s in pool if s._embedding is None]
        if not missing:
            return
        if self._embedding_fn is None:
            raise ValueError("No embedding_fn set — call set_embedding_column() first")
        embeddings = self._embedding_fn([s.get_data() for s in missing])
        for sample, embedding in zip(missing, embeddings):
            sample._set_embedding(np.asarray(embedding))

    def get_embeddings(self, samples: List[EvoSample]) -> np.ndarray:
        self.compute_embeddings(samples)
        return np.stack([s.get_embedding() for s in samples]) if samples else np.empty((0, 0))

    def get_embedding_matrix(
        self, status: Optional[Status] = None
    ) -> Tuple[np.ndarray, List[EvoSample]]:
        samples = self.get_by_status(status) if status is not None else self.get_all_samples()
        return self.get_embeddings(samples), samples

    def size(self) -> int:
        return len(self._by_id)

    def size_by_status(self) -> Dict[Status, int]:
        counts = {status: 0 for status in Status}
        for sample in self._by_id.values():
            counts[sample.get_status()] += 1
        return counts
