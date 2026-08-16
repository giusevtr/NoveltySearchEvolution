"""Tests for novelty_selection.score()."""

from __future__ import annotations

import numpy as np
import pytest

from novelty_search_evolution import novelty_selection


class TestScoreWithReference:
    def test_euclidean_farther_reference_scores_higher(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        reference = np.array([[0.0, 0.1], [0.0, 0.2]])
        scores = novelty_selection.score(X, k=1, distance="euclidean", reference=reference)
        assert scores[1] > scores[0]

    def test_k_clamped_to_reference_size(self):
        X = np.array([[0.0, 0.0]])
        reference = np.array([[1.0, 0.0], [2.0, 0.0]])
        # k much larger than the reference pool should not error
        scores = novelty_selection.score(X, k=100, distance="euclidean", reference=reference)
        expected = np.mean([1.0, 2.0])
        assert scores[0] == pytest.approx(expected)

    def test_cosine_distance(self):
        X = np.array([[1.0, 0.0]])
        reference = np.array([[1.0, 0.0], [0.0, 1.0]])
        scores = novelty_selection.score(X, k=1, distance="cosine", reference=reference)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)


class TestScoreSelfReference:
    def test_reference_none_excludes_self(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        scores = novelty_selection.score(X, k=1, distance="euclidean", reference=None)
        # point 0's nearest neighbor (excluding self) is point 1, distance 1
        assert scores[0] == pytest.approx(1.0)
        # point 2's nearest neighbor is point 1, distance 4
        assert scores[2] == pytest.approx(4.0)

    def test_k_clamped_when_pool_small(self):
        X = np.array([[0.0, 0.0], [3.0, 0.0]])
        scores = novelty_selection.score(X, k=10, distance="euclidean", reference=None)
        assert scores[0] == pytest.approx(3.0)


class TestInvalidDistance:
    def test_unknown_distance_raises(self):
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            novelty_selection.score(X, k=1, distance="manhattan", reference=None)


class TestInvalidInputs:
    def test_k_below_one_raises(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="k must be at least 1"):
            novelty_selection.score(X, k=0, distance="euclidean", reference=None)

    def test_empty_reference_raises(self):
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="reference is empty"):
            novelty_selection.score(X, k=1, distance="euclidean", reference=np.empty((0, 2)))

    def test_mismatched_reference_dimension_raises(self):
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="features"):
            novelty_selection.score(X, k=1, distance="euclidean", reference=np.zeros((2, 3)))

    def test_single_row_self_reference_raises(self):
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="single row"):
            novelty_selection.score(X, k=1, distance="euclidean", reference=None)

    def test_nan_distances_raise(self):
        X = np.array([[0.0, 0.0]])
        reference = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError, match="NaN"):
            novelty_selection.score(X, k=1, distance="cosine", reference=reference)

    def test_empty_X_returns_empty_scores(self):
        scores = novelty_selection.score(
            np.empty((0, 2)), k=1, distance="euclidean", reference=np.zeros((1, 2))
        )
        assert scores.shape == (0,)

    def test_non_2d_X_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            novelty_selection.score(np.array([1.0, 2.0]), k=1, distance="euclidean")
