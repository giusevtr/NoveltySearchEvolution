"""Tests for EvolutionEngine's crossover path and pluggable configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from novelty_search_evolution import evolution_engine as evolution_engine_module
from novelty_search_evolution.evolution_engine import EvolutionEngine
from novelty_search_evolution.population import Population
from novelty_search_evolution.sample import EvoSample, Status
from novelty_search_evolution.selection_engine import DefaultSelectionEngine


def embedding_fn(data_list):
    return [np.array([float(d)]) for d in data_list]


def make_engine(tmp_path, seeds, num_crossover_samples=2, selection_size=2):
    pop = Population()
    pop.set_embedding_column(embedding_fn)
    pop.set_seeds(seeds)

    engine = EvolutionEngine(pop, selection_size=selection_size, log_path=tmp_path)
    engine.set_filters([])
    engine.set_selection_config(
        {"nn-k": 1, "distance": "euclidean", "archive_update_prob": 0.0}
    )
    return pop, engine


class TestCrossoverParentPairs:
    def test_generates_one_pair_per_crossover_sample(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3])
        engine.set_crossover(lambda p1, p2: [], num_crossover_samples=5)

        pairs = engine._select_crossover_pairs()
        assert len(pairs) == 5

    def test_pairs_are_distinct_active_samples(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3])
        engine.set_crossover(lambda p1, p2: [], num_crossover_samples=10)

        active_ids = {s.get_id() for s in pop.get_active()}
        for p1, p2 in engine._select_crossover_pairs():
            assert p1.get_id() != p2.get_id()
            assert {p1.get_id(), p2.get_id()} <= active_ids

    @pytest.mark.parametrize("num_crossover_samples", [0, -1])
    def test_non_positive_crossover_samples_yields_no_pairs(
        self, tmp_path, num_crossover_samples
    ):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3])
        engine.set_crossover(lambda p1, p2: [], num_crossover_samples=num_crossover_samples)
        assert engine._select_crossover_pairs() == []

    def test_fewer_than_two_active_yields_no_pairs(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0])
        engine.set_crossover(lambda p1, p2: [], num_crossover_samples=3)
        assert engine._select_crossover_pairs() == []


class TestCrossoverCandidateGeneration:
    def test_crossover_children_record_both_parents(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=4)
        engine.set_crossover(lambda p1, p2: [p1.get_data() + p2.get_data()], 2)

        result = engine.step()
        children = [s for s in pop.get_all_samples() if s.get_parents()]

        assert result.num_candidates == 2
        assert len(children) == 2
        for child in children:
            assert len(child.get_parents()) == 2
            assert child.get_depth() == 1
            assert child.get_generation() == 0
            for parent in child.get_parents():
                assert child in parent.get_children()

    def test_crossover_fn_may_return_multiple_candidates(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=4)
        engine.set_crossover(lambda p1, p2: [100, 200, 300], num_crossover_samples=2)

        result = engine.step()
        assert result.num_candidates == 6
        assert result.num_accepted == 6

    def test_crossover_candidates_run_through_filters(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=4)
        engine.set_crossover(lambda p1, p2: [p1.get_data() + p2.get_data()], 2)
        engine.set_filters([lambda candidates: [{"accepted": False, "feedback": "no"}
                                                for _ in candidates]])

        result = engine.step()
        assert result.num_accepted == 0
        assert result.num_rejected == 2
        for child in (s for s in pop.get_all_samples() if s.get_parents()):
            assert child.get_status() == Status.REJECTED
            assert child.get_feedback() == ["no"]

    def test_crossover_logged_in_generation_trace(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=4)
        engine.set_crossover(lambda p1, p2: [p1.get_data() + p2.get_data()], 2)

        engine.step()
        log = (tmp_path / "generation_0_log.txt").read_text()
        assert "crossover pairs (2)" in log
        assert log.count("crossover: (") == 2

    def test_mutation_and_crossover_candidates_combine(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=4)
        engine.set_mutation(lambda p: [p.get_data() + 10], num_mutation_samples=3)
        engine.set_crossover(lambda p1, p2: [p1.get_data() + p2.get_data()], 2)

        result = engine.step()
        assert result.num_candidates == 5

    def test_no_crossover_fn_means_no_crossover_children(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=4)
        engine.set_mutation(lambda p: [p.get_data() + 10], num_mutation_samples=2)

        result = engine.step()
        for child in (s for s in pop.get_all_samples() if s.get_parents()):
            assert len(child.get_parents()) == 1
        assert result.num_candidates == 2


class TestSelectionEngineConfiguration:
    def test_default_engine_built_from_selection_config(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1])
        engine.set_selection_config({"nn-k": 4, "distance": "euclidean"})

        selection_engine = engine._get_selection_engine()
        assert isinstance(selection_engine, DefaultSelectionEngine)
        assert selection_engine.nn_k == 4
        assert selection_engine.distance == "euclidean"
        assert engine._get_selection_engine() is selection_engine  # cached

    def test_default_engine_defaults_to_cosine_distance(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1])
        engine.set_selection_config({"nn-k": 2})
        assert engine._get_selection_engine().distance == "cosine"

    def test_set_selection_engine_overrides_default(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=1)
        engine.set_mutation(lambda p: [p.get_data() + 10], num_mutation_samples=1)

        calls = []

        class FirstOnlySelectionEngine:
            def select(self, candidates, top_k):
                calls.append((len(candidates), top_k))
                return candidates[:top_k]

        engine.set_selection_engine(FirstOnlySelectionEngine())
        result = engine.step()

        assert engine._get_selection_engine().__class__ is FirstOnlySelectionEngine
        assert len(calls) == 1
        assert calls[0][1] == 1
        assert result.num_selected == 1


class TestDefaultLogDir:
    def test_resolves_to_directory_containing_pyproject(self):
        log_dir = evolution_engine_module._default_log_dir()
        assert log_dir.name == "logs"
        assert (log_dir.parent / "pyproject.toml").exists()

    def test_falls_back_to_relative_logs_dir(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert evolution_engine_module._default_log_dir() == Path("logs")


class TestStepWithoutOperators:
    def test_step_with_no_operators_recompetes_active_population(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0, 1, 2, 3], selection_size=2)

        result = engine.step()
        assert result.num_candidates == 0
        assert result.num_accepted == 0
        assert result.num_selected == 2
        assert len(pop.get_active()) == 2

    def test_step_on_empty_population_selects_nothing(self, tmp_path):
        pop = Population()
        pop.set_embedding_column(embedding_fn)
        engine = EvolutionEngine(pop, selection_size=2, log_path=tmp_path)
        engine.set_filters([])
        engine.set_selection_config({"nn-k": 1, "distance": "euclidean"})

        result = engine.step()
        assert (result.num_candidates, result.num_selected) == (0, 0)
        assert result.newly_active_ids == []
        assert result.newly_inactive_ids == []


class TestRevival:
    def test_inactive_samples_can_be_revived(self, tmp_path):
        pop, engine = make_engine(tmp_path, seeds=[0], selection_size=1)
        dormant = EvoSample(data=99)
        pop.add(dormant)
        dormant.set_inactive()

        # 99 is far from the archived seed 0, so novelty selection must prefer reviving it
        result = engine.step()
        assert result.newly_active_ids == [dormant.get_id()]
        assert dormant.get_status() == Status.ACTIVE
