"""EvoSample data model — a single unit in the evolving population.

`data` and `parents` never change after construction. `children`, `feedback`, `_status`,
and `_embedding` grow/are set over the sample's lifetime, but only via the methods below —
never by external direct assignment.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum
from typing import Any, List, Optional

import numpy as np


class Status(Enum):
    STALE = "stale"
    ACTIVE = "active"
    INACTIVE = "inactive"
    REJECTED = "rejected"


class EvoSample:
    def __init__(
        self,
        data: Any,
        parents: Optional[List["EvoSample"]] = None,
        generation: int = 0,
        id: Optional[str] = None,
    ) -> None:
        self.id: str = id or str(uuid.uuid4())
        self.data: Any = data
        self.parents: List["EvoSample"] = parents if parents is not None else []
        self.children: List["EvoSample"] = []
        self.generation: int = generation
        self.depth: int = 0 if not self.parents else max(p.depth for p in self.parents) + 1
        self.feedback: List[str] = []
        self._status: Status = Status.STALE
        self._embedding: Optional[np.ndarray] = None

    # --- accessors ---

    def get_id(self) -> str:
        return self.id

    def get_data(self) -> Any:
        return self.data

    def get_status(self) -> Status:
        return self._status

    def get_embedding(self) -> np.ndarray:
        if self._embedding is None:
            raise ValueError(f"Embedding not yet computed for sample {self.id}")
        return self._embedding

    def get_feedback(self) -> List[str]:
        return self.feedback

    def get_generation(self) -> int:
        return self.generation

    def get_depth(self) -> int:
        return self.depth

    # --- genealogy helpers ---

    def get_parents(self) -> List["EvoSample"]:
        return self.parents

    def get_children(self) -> List["EvoSample"]:
        return self.children

    def sample_accepted_children(self, k: Optional[int] = None) -> List["EvoSample"]:
        accepted = [c for c in self.children if c.get_status() != Status.REJECTED]
        return self._sample(accepted, k)

    def sample_rejected_children(self, k: Optional[int] = None) -> List["EvoSample"]:
        rejected = [c for c in self.children if c.get_status() == Status.REJECTED]
        return self._sample(rejected, k)

    @staticmethod
    def _sample(pool: List["EvoSample"], k: Optional[int]) -> List["EvoSample"]:
        if k is None or k >= len(pool):
            return list(pool)
        return random.sample(pool, k)

    # --- status transitions ---

    def set_active(self, feedback: Optional[str] = None) -> None:
        self._status = Status.ACTIVE
        if feedback is not None:
            self._add_feedback(feedback)

    def set_stale(self, feedback: Optional[str] = None) -> None:
        self._status = Status.STALE
        if feedback is not None:
            self._add_feedback(feedback)

    def set_inactive(self, feedback: Optional[str] = None) -> None:
        self._status = Status.INACTIVE
        if feedback is not None:
            self._add_feedback(feedback)

    def set_reject(self, feedback: Optional[str] = None) -> None:
        self._status = Status.REJECTED
        if feedback is not None:
            self._add_feedback(feedback)

    # --- internal ---

    def _add_child(self, child: "EvoSample") -> None:
        self.children.append(child)

    def _add_feedback(self, feedback: str) -> None:
        self.feedback.append(feedback)

    def _set_embedding(self, embedding: np.ndarray) -> None:
        if self._embedding is not None:
            raise ValueError(f"Embedding already set for sample {self.id}")
        self._embedding = embedding

    # --- equality/hash by id ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EvoSample):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"EvoSample(id={self.id!r}, status={self._status.value}, generation={self.generation})"
