"""EvolutionEngine — drives the step() algorithm (SPEC §4, §5)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .population import Population
from .population_viewer import render_population_html
from .sample import EvoSample, Status
from .selection_engine import DefaultSelectionEngine, SelectionEngine

MutFn = Callable[[EvoSample], List[Any]]
CrossoverFn = Callable[[EvoSample, EvoSample], List[Any]]
FilterFn = Callable[[List[EvoSample]], List[Dict[str, Any]]]


def _default_log_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "logs"
    return Path("logs")


@dataclass
class StepResult:
    generation: int
    num_candidates: int
    num_accepted: int
    num_rejected: int
    num_selected: int
    newly_active_ids: List[str] = field(default_factory=list)
    newly_inactive_ids: List[str] = field(default_factory=list)
    newly_archived_ids: List[str] = field(default_factory=list)


class EvolutionEngine:
    def __init__(
        self,
        population: Population,
        selection_size: int,
        log_path: Optional[Union[Path, str]] = None,
    ) -> None:
        self.population = population
        self._generation = 0
        self._selection_size = selection_size
        self._log_path = Path(log_path) if log_path is not None else _default_log_dir()
        self._log_path.mkdir(parents=True, exist_ok=True)

        self._mut_fn: Optional[MutFn] = None
        self._num_mutation_samples = 0
        self._crossover_fn: Optional[CrossoverFn] = None
        self._num_crossover_samples = 0
        self._filter_fns: List[FilterFn] = []
        self._selection_config: Dict[str, Any] = {}
        self._selection_engine: Optional[SelectionEngine] = None

    # --- setup ---

    def set_mutation(self, mut_fn: MutFn, num_mutation_samples: int) -> None:
        self._mut_fn = mut_fn
        self._num_mutation_samples = num_mutation_samples

    def set_crossover(self, crossover_fn: CrossoverFn, num_crossover_samples: int) -> None:
        self._crossover_fn = crossover_fn
        self._num_crossover_samples = num_crossover_samples

    def set_filters(self, filter_fns: List[FilterFn]) -> None:
        self._filter_fns = filter_fns

    def set_selection_config(self, config: Dict[str, Any]) -> None:
        self._selection_config = config

    def set_selection_engine(self, engine: SelectionEngine) -> None:
        self._selection_engine = engine

    def _get_selection_engine(self) -> SelectionEngine:
        if self._selection_engine is None:
            self._selection_engine = DefaultSelectionEngine(
                self.population,
                nn_k=self._selection_config["nn-k"],
                distance=self._selection_config.get("distance", "cosine"),
            )
        return self._selection_engine

    # --- parent selection (§4.3) ---

    def _select_mutation_parents(self) -> List[EvoSample]:
        return self.population.sample_random(self._num_mutation_samples, status=Status.ACTIVE)

    def _select_crossover_pairs(self) -> List[Tuple[EvoSample, EvoSample]]:
        active = self.population.get_active()
        if len(active) < 2 or self._num_crossover_samples <= 0:
            return []

        pairs: List[Tuple[EvoSample, EvoSample]] = []
        for _ in range(self._num_crossover_samples):
            p1, p2 = random.sample(active, 2)
            pairs.append((p1, p2))
        return pairs

    # --- step algorithm (§5) ---

    def step(self) -> StepResult:
        generation = self._generation
        trace: List[str] = [f"=== Generation {generation} ==="]

        # 1. parent selection
        mut_fn = self._mut_fn
        crossover_fn = self._crossover_fn
        mutation_parents = self._select_mutation_parents() if mut_fn else []
        crossover_pairs = self._select_crossover_pairs() if crossover_fn else []

        trace.append("--- 1. parent selection ---")
        trace.append(
            f"mutation parents ({len(mutation_parents)}): "
            f"{[p.get_id() for p in mutation_parents]}"
        )
        trace.append(
            f"crossover pairs ({len(crossover_pairs)}): "
            f"{[(p1.get_id(), p2.get_id()) for p1, p2 in crossover_pairs]}"
        )

        # 2. generate candidates
        candidates: List[EvoSample] = []
        if mut_fn is not None:
            for parent in mutation_parents:
                for raw in mut_fn(parent):
                    candidate = EvoSample(data=raw, parents=[parent], generation=generation)
                    parent._add_child(candidate)
                    candidates.append(candidate)
                    trace.append(f"mutation: {parent.get_id()} -> {candidate.get_id()}")
        if crossover_fn is not None:
            for p1, p2 in crossover_pairs:
                for raw in crossover_fn(p1, p2):
                    candidate = EvoSample(data=raw, parents=[p1, p2], generation=generation)
                    p1._add_child(candidate)
                    p2._add_child(candidate)
                    candidates.append(candidate)
                    trace.append(
                        f"crossover: ({p1.get_id()}, {p2.get_id()}) -> {candidate.get_id()}"
                    )
        trace.append(f"total candidates generated: {len(candidates)}")

        # 3. filter
        trace.append("--- 2. filtering ---")
        for filter_fn in self._filter_fns:
            unrejected = [c for c in candidates if c.get_status() != Status.REJECTED]
            if not unrejected:
                trace.append(f"filter {filter_fn.__name__}: skipped (no unrejected candidates)")
                break
            judgements = filter_fn(unrejected)
            num_rejected_by_filter = 0
            for candidate, judgement in zip(unrejected, judgements):
                if not judgement.get("accepted", False):
                    candidate.set_reject(feedback=judgement.get("feedback") or None)
                    num_rejected_by_filter += 1
                    trace.append(
                        f"  rejected {candidate.get_id()}: {judgement.get('feedback') or ''}"
                    )
                elif judgement.get("feedback"):
                    candidate._add_feedback(judgement["feedback"])
            trace.append(
                f"filter {filter_fn.__name__}: evaluated {len(unrejected)}, "
                f"rejected {num_rejected_by_filter}"
            )

        for candidate in candidates:
            self.population.add(candidate)
            if candidate.get_status() != Status.REJECTED:
                candidate.set_stale()

        num_rejected = sum(1 for c in candidates if c.get_status() == Status.REJECTED)
        accepted_candidates = [c for c in candidates if c.get_status() != Status.REJECTED]
        trace.append(f"accepted: {len(accepted_candidates)}, rejected: {num_rejected}")

        # 4. select
        trace.append("--- 3. selection ---")
        previously_active = self.population.get_active()
        top_k = self._selection_size

        accepted_ids = {c.get_id() for c in accepted_candidates}
        dormant_pool = [
            s
            for s in self.population.get_stale() + self.population.get_inactive()
            if s.get_id() not in accepted_ids
        ]
        num_revival = min(top_k, len(dormant_pool))
        revival_sample = random.sample(dormant_pool, num_revival) if num_revival else []

        selection_pool = list(accepted_candidates) + list(previously_active) + revival_sample
        trace.append(
            f"selection pool: {len(accepted_candidates)} accepted + "
            f"{len(previously_active)} previously active + "
            f"{len(revival_sample)} revived = {len(selection_pool)} total "
            f"(selection_size={top_k})"
        )

        selected: List[EvoSample] = []
        if selection_pool:
            self.population.compute_embeddings(selection_pool)
            selection_engine = self._get_selection_engine()
            selected = selection_engine.select(selection_pool, top_k)

        selected_ids = {s.get_id() for s in selected}
        for s in selected:
            s.set_active()
        newly_inactive_ids = []
        for s in previously_active:
            if s.get_id() not in selected_ids:
                s.set_inactive()
                newly_inactive_ids.append(s.get_id())
        trace.append(f"selected ({len(selected)}): {sorted(selected_ids)}")
        trace.append(f"newly inactive ({len(newly_inactive_ids)}): {newly_inactive_ids}")

        # 5. archive update
        trace.append("--- 4. archive update ---")
        archive_update_prob = self._selection_config.get("archive_update_prob", 0.1)
        newly_archived_ids = []
        for candidate in accepted_candidates:
            if random.random() < archive_update_prob:
                self.population.add_to_archive(candidate)
                newly_archived_ids.append(candidate.get_id())
        trace.append(
            f"archive_update_prob={archive_update_prob}, "
            f"newly archived ({len(newly_archived_ids)}): {newly_archived_ids}"
        )

        self._generation += 1

        log_file = self._log_path / f"generation_{generation}_log.txt"
        log_file.write_text("\n".join(trace) + "\n")
        self._snapshot_population(generation)

        return StepResult(
            generation=generation,
            num_candidates=len(candidates),
            num_accepted=len(accepted_candidates),
            num_rejected=num_rejected,
            num_selected=len(selected),
            newly_active_ids=[s.get_id() for s in selected],
            newly_inactive_ids=newly_inactive_ids,
            newly_archived_ids=newly_archived_ids,
        )

    def _snapshot_population(self, generation: int) -> None:
        snapshot = [
            {
                "id": s.get_id(),
                "data": s.get_data(),
                "status": s.get_status().value,
                "generation": s.get_generation(),
                "depth": s.get_depth(),
                "parent_ids": [p.get_id() for p in s.get_parents()],
                "child_ids": [c.get_id() for c in s.get_children()],
                "feedback": s.get_feedback(),
            }
            for s in self.population.get_all_samples()
        ]
        snapshot_dir = self._log_path / "population_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = snapshot_dir / "population.json"
        snapshot_file.write_text(json.dumps(snapshot, indent=2, default=str))

        viewer_file = snapshot_dir / "viewer.html"
        viewer_file.write_text(render_population_html(snapshot))

    def run(self, num_generations: int) -> List[StepResult]:
        results = []
        for _ in range(num_generations):
            result = self.step()
            results.append(result)
        return results
