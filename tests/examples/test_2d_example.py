"""Tests for examples/2d_example/run.py — obstacle geometry and evolution operators.

The example lives in a directory whose name is not a valid Python identifier, so it is loaded
by path. matplotlib is an optional (examples-only) dependency, so the module is skipped when
it is unavailable.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="examples extra (matplotlib) not installed")

_RUN_PY = Path(__file__).resolve().parents[2] / "examples" / "2d_example" / "run.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("two_d_example_run", _RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()


class FakeSample:
    """Minimal stand-in for EvoSample: the operators only ever call get_data()."""

    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


class TestInCircle:
    @pytest.mark.parametrize(
        "point, expected",
        [
            ((0.5, 0.5), True),  # center
            ((0.6, 0.5), True),  # exactly on the boundary
            ((0.5, 0.6), True),
            ((0.61, 0.5), False),  # just outside
            ((1.0, 1.0), False),
        ],
    )
    def test_membership(self, point, expected):
        obstacle = {"center": (0.5, 0.5), "radius": 0.1}
        assert example.in_circle(point, obstacle) is expected


class TestInRect:
    @pytest.mark.parametrize(
        "point, expected",
        [
            ((0.5, 0.5), True),  # interior
            ((0.4, 0.4), True),  # corner, inclusive
            ((0.6, 0.6), True),  # opposite corner, inclusive
            ((0.39, 0.5), False),
            ((0.5, 0.61), False),
        ],
    )
    def test_membership(self, point, expected):
        obstacle = {"x_min": 0.4, "x_max": 0.6, "y_min": 0.4, "y_max": 0.6}
        assert example.in_rect(point, obstacle) is expected


class TestBlockingObstacle:
    def test_free_point_is_unblocked(self):
        assert example.blocking_obstacle((0.02, 0.02)) is None

    def test_reports_circle_index(self):
        center = example.CIRCLE_OBSTACLES[1]["center"]
        assert example.blocking_obstacle(center) == "inside circle #1"

    def test_reports_rectangle_index(self):
        rect = example.RECT_OBSTACLES[0]
        middle = (
            (rect["x_min"] + rect["x_max"]) / 2,
            (rect["y_min"] + rect["y_max"]) / 2,
        )
        assert example.blocking_obstacle(middle) == "inside rectangle #0"

    def test_every_obstacle_center_is_blocked(self):
        for obstacle in example.CIRCLE_OBSTACLES:
            assert example.blocking_obstacle(obstacle["center"]) is not None
        for rect in example.RECT_OBSTACLES:
            middle = (
                (rect["x_min"] + rect["x_max"]) / 2,
                (rect["y_min"] + rect["y_max"]) / 2,
            )
            assert example.blocking_obstacle(middle) is not None


class TestRandomValidPoint:
    def test_returns_point_outside_every_obstacle(self):
        random.seed(0)
        for _ in range(50):
            point = example.random_valid_point()
            assert example.blocking_obstacle(point) is None
            assert 0.0 <= point[0] <= 1.0
            assert 0.0 <= point[1] <= 1.0

    def test_retries_until_a_free_point_is_drawn(self, monkeypatch):
        blocked = example.CIRCLE_OBSTACLES[0]["center"]
        draws = iter([blocked[0], blocked[1], 0.02, 0.02])
        monkeypatch.setattr(example.random, "uniform", lambda a, b: next(draws))

        assert example.random_valid_point() == (0.02, 0.02)


class TestMutFn:
    def test_returns_configured_number_of_candidates(self):
        random.seed(1)
        candidates = example.mut_fn(FakeSample((0.5, 0.5)))
        assert len(candidates) == example.MUTATIONS_PER_PARENT

    def test_candidates_stay_inside_unit_square(self):
        random.seed(2)
        for parent in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]:
            for x, y in example.mut_fn(FakeSample(parent)):
                assert 0.0 <= x <= 1.0
                assert 0.0 <= y <= 1.0

    def test_zero_noise_reproduces_parent(self, monkeypatch):
        monkeypatch.setattr(example.random, "gauss", lambda mu, sigma: 0.0)
        assert example.mut_fn(FakeSample((0.25, 0.75))) == [
            (0.25, 0.75)
        ] * example.MUTATIONS_PER_PARENT

    def test_returns_plain_floats(self):
        random.seed(3)
        for x, y in example.mut_fn(FakeSample((0.5, 0.5))):
            assert type(x) is float
            assert type(y) is float


class TestCrossoverFn:
    def test_zero_jitter_returns_parent_midpoint(self, monkeypatch):
        monkeypatch.setattr(example.random, "gauss", lambda mu, sigma: 0.0)
        child = example.crossover_fn(FakeSample((0.2, 0.4)), FakeSample((0.6, 0.8)))
        assert child == [(0.4, pytest.approx(0.6))]

    def test_returns_single_candidate_inside_unit_square(self):
        random.seed(4)
        child = example.crossover_fn(FakeSample((0.0, 0.0)), FakeSample((1.0, 1.0)))
        assert len(child) == 1
        x, y = child[0]
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0

    def test_clips_out_of_range_jitter(self, monkeypatch):
        monkeypatch.setattr(example.random, "gauss", lambda mu, sigma: 5.0)
        assert example.crossover_fn(FakeSample((1.0, 1.0)), FakeSample((1.0, 1.0))) == [
            (1.0, 1.0)
        ]


class TestFilterFn:
    def test_accepts_free_points_and_rejects_blocked_ones(self):
        free = FakeSample((0.02, 0.02))
        blocked = FakeSample(example.CIRCLE_OBSTACLES[0]["center"])

        judgements = example.filter_fn([free, blocked])
        assert judgements[0] == {"accepted": True, "feedback": ""}
        assert judgements[1]["accepted"] is False
        assert judgements[1]["feedback"] == "inside circle #0"

    def test_one_judgement_per_candidate_in_order(self):
        candidates = [FakeSample((0.02, 0.02)), FakeSample((0.03, 0.03))]
        assert len(example.filter_fn(candidates)) == len(candidates)

    def test_empty_candidate_list(self):
        assert example.filter_fn([]) == []


class TestEmbeddingFn:
    def test_embeds_each_point_as_a_2d_vector(self):
        embeddings = example.embedding_fn([(0.1, 0.2), (0.3, 0.4)])
        assert len(embeddings) == 2
        assert all(isinstance(e, np.ndarray) and e.shape == (2,) for e in embeddings)
        assert embeddings[0].tolist() == [0.1, 0.2]

    def test_empty_batch(self):
        assert example.embedding_fn([]) == []


class TestPlotting:
    def test_plot_generation_writes_png_for_a_step(self, tmp_path, monkeypatch):
        from novelty_search_evolution import EvolutionEngine, Population

        monkeypatch.setattr(example, "OUTPUT_DIR", tmp_path / "output")

        population = Population()
        population.set_seeds([(0.02, 0.02), (0.05, 0.05), (0.9, 0.9)])
        population.set_embedding_column(example.embedding_fn)

        engine = EvolutionEngine(population, selection_size=2, log_path=tmp_path / "logs")
        engine.set_mutation(example.mut_fn, num_mutation_candidates=2)
        engine.set_crossover(example.crossover_fn, num_crossover_samples=1)
        engine.set_filters([example.filter_fn])
        engine.set_selection_config(example.SELECTION_CONFIG)

        result = engine.step()
        example.plot_generation(result.generation, population, result)
        example.plot_generation(None, population, result)

        assert (tmp_path / "output" / "gen_000.png").exists()
        assert (tmp_path / "output" / "gen_seed.png").exists()

    def test_plot_generation_handles_empty_population(self, tmp_path, monkeypatch):
        from novelty_search_evolution import Population, StepResult

        monkeypatch.setattr(example, "OUTPUT_DIR", tmp_path / "output")
        result = StepResult(
            generation=0,
            num_candidates=0,
            num_accepted=0,
            num_rejected=0,
            num_selected=0,
        )
        example.plot_generation(0, Population(), result)
        assert (tmp_path / "output" / "gen_000.png").exists()
