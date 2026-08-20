"""Tests for EvolutionEngine.step()/run()."""

from __future__ import annotations

import json
import functools
import tempfile

import numpy as np
import pytest

from novelty_search_evolution.evolution_engine import EvolutionEngine
from novelty_search_evolution.population import Population
from novelty_search_evolution.sample import Status
from novelty_search_evolution.selection_engine import DefaultSelectionEngine


def embedding_fn(data_list):
    return [np.array([float(d)]) for d in data_list]


def make_engine(seeds, num_mutation_candidates=2, num_crossover_samples=0, selection_size=2,
                 archive_update_prob=0.0, filters=None, log_path=None):
    pop = Population()
    pop.set_embedding_column(embedding_fn)
    pop.set_seeds(seeds)

    if log_path is None:
        log_path = tempfile.mkdtemp()
    engine = EvolutionEngine(pop, selection_size=selection_size, log_path=log_path)
    counter = {"n": max(seeds) + 1 if seeds else 0}

    def mut_fn(parent):
        counter["n"] += 1
        return [counter["n"]]

    def crossover_fn(p1, p2):
        counter["n"] += 1
        return [counter["n"]]

    engine.set_mutation(mut_fn, num_mutation_candidates=num_mutation_candidates)
    engine.set_crossover(crossover_fn, num_crossover_samples=num_crossover_samples)
    engine.set_filters(filters if filters is not None else [])
    engine.set_selection_config({
        "nn-k": 1,
        "distance": "euclidean",
        "archive_update_prob": archive_update_prob,
    })
    return pop, engine


class TestBasicStep:
    def test_step_generates_and_selects_candidates(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], num_mutation_candidates=2, selection_size=2)
        result = engine.step()

        assert result.generation == 0
        assert result.num_candidates == 2  # 2 mutation parents, 1 child each
        assert result.num_accepted == 2
        assert result.num_rejected == 0
        assert result.num_selected == 2

    def test_generation_increments(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3])
        engine.step()
        result2 = engine.step()
        assert result2.generation == 1

    def test_crossover_skipped_below_two_active(self):
        pop = Population()
        pop.set_embedding_column(embedding_fn)
        pop.set_seeds([0])  # only one active sample

        engine = EvolutionEngine(pop, selection_size=1, log_path=tempfile.mkdtemp())
        engine.set_mutation(lambda p: [], num_mutation_candidates=0)
        calls = []

        def crossover_fn(p1, p2):
            calls.append((p1, p2))
            return [99]

        engine.set_crossover(crossover_fn, num_crossover_samples=2)
        engine.set_filters([])
        engine.set_selection_config({"nn-k": 1, "distance": "euclidean",
                                      "archive_update_prob": 0.0})
        engine.step()
        assert calls == []


class TestFilters:
    def test_rejected_candidates_marked_and_excluded(self):
        def reject_all(candidates):
            return [{"accepted": False, "feedback": "no"} for _ in candidates]

        pop, engine = make_engine(seeds=[0, 1, 2, 3], filters=[reject_all], selection_size=2)
        result = engine.step()

        assert result.num_accepted == 0
        assert result.num_rejected == result.num_candidates
        # With no accepted candidates, the select step still re-competes the
        # previously-active population (revival/reconfirmation dynamics), so
        # selection is not necessarily empty.
        for s in pop.get_rejected():
            assert s.get_status() == Status.REJECTED

    def test_short_circuit_skips_later_filters(self):
        first_calls = []
        second_calls = []

        def filter_reject(candidates):
            first_calls.append(len(candidates))
            return [{"accepted": False, "feedback": "rejected"} for _ in candidates]

        def filter_never_reached(candidates):
            second_calls.append(len(candidates))
            return [{"accepted": True, "feedback": ""} for _ in candidates]

        pop, engine = make_engine(
            seeds=[0, 1, 2, 3], filters=[filter_reject, filter_never_reached], selection_size=2
        )
        engine.step()
        assert first_calls  # ran
        assert second_calls == []  # short-circuited, never invoked

    def test_feedback_recorded_on_accept(self):
        def note_filter(candidates):
            return [{"accepted": True, "feedback": "looks good"} for _ in candidates]

        pop, engine = make_engine(seeds=[0, 1, 2, 3], filters=[note_filter], selection_size=2)
        engine.step()
        accepted = [s for s in pop.get_all_samples()
                    if s.get_parents() and s.get_status() != Status.REJECTED]
        for s in accepted:
            assert "looks good" in s.get_feedback()


class TestActiveInactiveTransitions:
    def test_selected_become_active_others_inactive(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], num_mutation_candidates=4, selection_size=2)
        previously_active_ids = {s.get_id() for s in pop.get_active()}
        result = engine.step()

        assert len(result.newly_active_ids) == 2
        # seeds not selected should now be inactive
        newly_inactive = set(result.newly_inactive_ids)
        assert newly_inactive.issubset(previously_active_ids)
        assert len(pop.get_active()) == 2


class TestArchiveUpdate:
    def test_archive_update_prob_zero_never_archives_new(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], archive_update_prob=0.0, selection_size=2)
        initial_archive_size = len(pop.get_archive())
        engine.step()
        assert len(pop.get_archive()) == initial_archive_size

    def test_archive_update_prob_one_archives_all_accepted(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], archive_update_prob=1.0, selection_size=2)
        result = engine.step()
        assert len(result.newly_archived_ids) == result.num_accepted


class TestRun:
    def test_run_returns_n_results(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], selection_size=2)
        results = engine.run(3)
        assert len(results) == 3
        assert [r.generation for r in results] == [0, 1, 2]

    def test_run_overwrites_population_snapshot_each_generation(self, tmp_path):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], selection_size=2, log_path=tmp_path)
        engine.run(3)

        snapshot_dir = tmp_path / "population_snapshot"
        snapshot_file = snapshot_dir / "population.json"
        assert snapshot_file.exists()
        # only the latest generation's snapshot is kept — no per-generation files
        assert list(snapshot_dir.glob("generation_*_population.json")) == []

        snapshot = json.loads(snapshot_file.read_text())
        sample = snapshot[0]
        assert set(sample.keys()) == {
            "id", "data", "status", "generation", "depth",
            "parent_ids", "child_ids", "feedback",
        }

    def test_run_overwrites_viewer_html_each_generation(self, tmp_path):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], selection_size=2, log_path=tmp_path)
        engine.run(3)

        snapshot_dir = tmp_path / "population_snapshot"
        viewer_file = snapshot_dir / "viewer.html"
        assert viewer_file.exists()
        assert viewer_file.read_text().strip().startswith("<!doctype html>")
        assert list(snapshot_dir.glob("generation_*_viewer.html")) == []


class TestLogging:
    def test_default_log_path_resolves_to_project_root_logs_dir(self):
        pop = Population()
        pop.set_embedding_column(embedding_fn)
        pop.set_seeds([0, 1])

        engine = EvolutionEngine(pop, selection_size=1)
        assert engine._log_path.name == "logs"
        assert (engine._log_path / "..").resolve().exists()

    def test_step_writes_generation_log_file(self, tmp_path):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], selection_size=2, log_path=tmp_path)
        engine.step()

        log_file = tmp_path / "generation_0_log.txt"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Generation 0" in content
        assert "parent selection" in content
        assert "filtering" in content
        assert "selection" in content
        assert "archive update" in content

    def test_log_file_named_per_generation(self, tmp_path):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], selection_size=2, log_path=tmp_path)
        engine.run(3)

        for i in range(3):
            assert (tmp_path / f"generation_{i}_log.txt").exists()


class TestCallbackContractValidation:
    def test_mut_fn_returning_none_raises(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3])
        engine.set_mutation(lambda parent: None, num_mutation_candidates=1)
        with pytest.raises(TypeError, match="mut_fn"):
            engine.step()

    def test_crossover_fn_returning_scalar_raises(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], num_crossover_samples=1)
        engine.set_mutation(lambda parent: [], num_mutation_candidates=0)
        engine.set_crossover(lambda p1, p2: 5, num_crossover_samples=1)
        with pytest.raises(TypeError, match="crossover_fn"):
            engine.step()

    def test_filter_judgement_missing_accepted_key_raises(self):
        def typo_filter(candidates):
            return [{"acepted": True} for _ in candidates]

        pop, engine = make_engine(seeds=[0, 1, 2, 3], filters=[typo_filter])
        with pytest.raises(ValueError, match="'accepted' key"):
            engine.step()

    def test_filter_returning_non_mapping_judgement_raises(self):
        def bad_filter(candidates):
            return [True for _ in candidates]

        pop, engine = make_engine(seeds=[0, 1, 2, 3], filters=[bad_filter])
        with pytest.raises(TypeError, match="must be a dict"):
            engine.step()

    def test_callable_filter_object_is_named_in_errors(self):
        class RejectAll:
            def __call__(self, candidates):
                return []

        pop, engine = make_engine(seeds=[0, 1, 2, 3], filters=[RejectAll()])
        with pytest.raises(ValueError, match="RejectAll"):
            engine.step()

    def test_missing_nn_k_raises_explanatory_error(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3])
        engine.set_selection_config({"distance": "euclidean", "archive_update_prob": 0.0})
        with pytest.raises(ValueError, match="nn-k"):
            engine.step()


class TestFilterContract:
    def test_judgement_count_mismatch_raises(self):
        def short_filter(candidates):
            return [{"accepted": True}]

        pop, engine = make_engine(seeds=[0, 1, 2, 3], filters=[short_filter], selection_size=2)
        with pytest.raises(ValueError, match="one judgement per candidate"):
            engine.step()

    def test_partial_filter_is_named_in_trace(self, tmp_path):
        def reject_all(candidates, reason):
            return [{"accepted": False, "feedback": reason} for _ in candidates]

        filter_fn = functools.partial(reject_all, reason="nope")
        pop, engine = make_engine(
            seeds=[0, 1, 2, 3], filters=[filter_fn], selection_size=2, log_path=tmp_path
        )
        result = engine.step()
        assert result.num_rejected == result.num_candidates
        assert "reject_all" in (tmp_path / "generation_0_log.txt").read_text()


class TestThinPopulation:
    def test_step_survives_empty_active_population(self):
        pop, engine = make_engine(seeds=[0, 1], num_mutation_candidates=5, selection_size=2)
        for s in pop.get_active():
            s.set_inactive()
        result = engine.step()
        assert result.num_candidates == 0

    def test_mutation_parents_sampled_with_replacement(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], num_mutation_candidates=10)
        active_ids = {p.get_id() for p in pop.get_active()}
        parents = engine._select_mutation_parents()
        assert len(parents) == 10
        assert all(p.get_id() in active_ids for p in parents)


class TestSelectionConfigReconfiguration:
    def test_config_change_rebuilds_default_engine(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], selection_size=2)
        engine.step()
        assert engine._get_selection_engine().nn_k == 1

        engine.set_selection_config({"nn-k": 3, "distance": "euclidean", "archive_update_prob": 0.0})
        assert engine._get_selection_engine().nn_k == 3

    def test_config_change_keeps_custom_engine(self):
        pop, engine = make_engine(seeds=[0, 1, 2, 3], selection_size=2)
        custom = DefaultSelectionEngine(pop, nn_k=2, distance="euclidean")
        engine.set_selection_engine(custom)
        engine.set_selection_config({"nn-k": 9, "distance": "euclidean"})
        assert engine._get_selection_engine() is custom
