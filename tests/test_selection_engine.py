"""Tests for DefaultSelectionEngine's greedy, archive-aware novelty selection."""

from __future__ import annotations

import numpy as np
import pytest

from novelty_search_evolution.population import Population
from novelty_search_evolution.sample import EvoSample
from novelty_search_evolution.selection_engine import DefaultSelectionEngine


def embedding_fn(data_list):
    return [np.asarray(d, dtype=float) for d in data_list]


def make_population(seeds=None):
    pop = Population()
    pop.set_embedding_column(embedding_fn)
    if seeds:
        pop.set_seeds(seeds)
    return pop


def make_candidates(pop, points):
    candidates = [EvoSample(data=p) for p in points]
    pop.bulk_add(candidates)
    return candidates


class TestDegenerateInputs:
    def test_empty_candidates_returns_empty(self):
        engine = DefaultSelectionEngine(make_population(), nn_k=1, distance="euclidean")
        assert engine.select([], top_k=3) == []

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_non_positive_top_k_returns_empty(self, top_k):
        pop = make_population()
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        candidates = make_candidates(pop, [[0.0], [1.0]])
        assert engine.select(candidates, top_k=top_k) == []

    def test_top_k_larger_than_candidates_returns_all(self):
        pop = make_population()
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        candidates = make_candidates(pop, [[0.0], [1.0], [2.0]])
        selected = engine.select(candidates, top_k=10)
        assert sorted(s.get_id() for s in selected) == sorted(c.get_id() for c in candidates)

    def test_selection_does_not_mutate_caller_list(self):
        pop = make_population()
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        candidates = make_candidates(pop, [[0.0], [1.0], [2.0]])
        engine.select(candidates, top_k=2)
        assert len(candidates) == 3


class TestArchiveAwareSelection:
    def test_picks_candidate_farthest_from_archive_first(self):
        pop = make_population(seeds=[[0.0]])
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        near, far = make_candidates(pop, [[0.1], [5.0]])

        selected = engine.select([near, far], top_k=1)
        assert selected == [far]

    def test_empty_archive_still_selects_via_self_distances(self):
        pop = make_population()  # no seeds -> empty archive
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        clustered_a, clustered_b, outlier = make_candidates(pop, [[0.0], [0.05], [9.0]])

        selected = engine.select([clustered_a, clustered_b, outlier], top_k=1)
        assert selected == [outlier]


class TestGreedyDiversity:
    def test_second_pick_avoids_near_duplicate_of_first(self):
        pop = make_population(seeds=[[0.0]])
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        # 10.01 is the most novel w.r.t. the archived 0.0; 10.0 is its near-duplicate, so the
        # second pick must be the diverse alternative (4.0) rather than the runner-up by score.
        near_duplicate, most_novel, middle = make_candidates(pop, [[10.0], [10.01], [4.0]])

        selected = engine.select([near_duplicate, most_novel, middle], top_k=2)
        assert selected == [most_novel, middle]

    def test_returns_candidates_in_selection_order(self):
        pop = make_population(seeds=[[0.0]])
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        candidates = make_candidates(pop, [[1.0], [3.0], [8.0]])

        selected = engine.select(candidates, top_k=3)
        assert [s.get_data() for s in selected] == [[8.0], [3.0], [1.0]]


class TestDistanceMetrics:
    def test_cosine_is_the_default_metric(self):
        pop = make_population()
        engine = DefaultSelectionEngine(pop, nn_k=1)
        assert engine.distance == "cosine"

    def test_cosine_ignores_magnitude(self):
        pop = make_population(seeds=[[1.0, 0.0]])
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="cosine")
        # same direction as the archived seed (distance 0) vs. orthogonal (distance 1)
        same_direction, orthogonal = make_candidates(pop, [[50.0, 0.0], [0.0, 1.0]])

        selected = engine.select([same_direction, orthogonal], top_k=1)
        assert selected == [orthogonal]

    def test_unknown_metric_rejected_by_scoring(self):
        pop = make_population(seeds=[[0.0]])
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="manhattan")
        candidates = make_candidates(pop, [[1.0]])
        with pytest.raises(ValueError, match="Unknown distance metric"):
            engine.select(candidates, top_k=1)


class TestArchiveIsNotModified:
    def test_select_leaves_population_archive_untouched(self):
        pop = make_population(seeds=[[0.0]])
        engine = DefaultSelectionEngine(pop, nn_k=1, distance="euclidean")
        candidates = make_candidates(pop, [[1.0], [2.0]])

        engine.select(candidates, top_k=2)
        assert [s.get_data() for s in pop.get_archive()] == [[0.0]]
