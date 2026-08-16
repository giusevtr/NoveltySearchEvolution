"""2D novelty search: cover [0,1]x[0,1] while avoiding obstacles.

Each generation is rendered to examples/2d_example/output/gen_{i:03d}.png so progress
(coverage spreading, obstacles being avoided) can be inspected visually.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from novelty_search_evolution import EvolutionEngine, Population, StepResult

# --- config ---

NUM_GENERATIONS = 30
NUM_SEEDS = 3
NUM_MUTATION_SAMPLES = 15
NUM_CROSSOVER_SAMPLES = 5
MUTATIONS_PER_PARENT = 3
MUTATION_SIGMA = 0.08
CROSSOVER_JITTER = 0.03

SELECTION_CONFIG = {
    "nn-k": 5,
    "distance": "euclidean",
    "archive_update_prob": 0.3,
}

SELECTION_SIZE = 10

OUTPUT_DIR = Path(__file__).parent / "output"

Point = Tuple[float, float]

CIRCLE_OBSTACLES = [
    {"center": (0.3, 0.7), "radius": 0.12},
    {"center": (0.75, 0.25), "radius": 0.15},
]
RECT_OBSTACLES = [
    {"x_min": 0.45, "x_max": 0.65, "y_min": 0.45, "y_max": 0.65},
    {"x_min": 0.0, "x_max": 0.15, "y_min": 0.65, "y_max": 0.75},
    {"x_min": 0.25, "x_max": 0.35, "y_min": 0.0, "y_max": 0.50},
    {"x_min": 0.7, "x_max": 1.00, "y_min": 0.5, "y_max": 0.55},
    {"x_min": 0.25, "x_max": 0.35, "y_min": 0.85, "y_max": 1.00},
    {"x_min": 0.7, "x_max": 0.80, "y_min": 0.6, "y_max": 1.0},
]


# --- obstacle geometry ---


def in_circle(point: Point, obstacle: dict) -> bool:
    cx, cy = obstacle["center"]
    x, y = point
    return (x - cx) ** 2 + (y - cy) ** 2 <= obstacle["radius"] ** 2


def in_rect(point: Point, obstacle: dict) -> bool:
    x, y = point
    return obstacle["x_min"] <= x <= obstacle["x_max"] and obstacle["y_min"] <= y <= obstacle["y_max"]


def blocking_obstacle(point: Point) -> str | None:
    for i, obstacle in enumerate(CIRCLE_OBSTACLES):
        if in_circle(point, obstacle):
            return f"inside circle #{i}"
    for i, obstacle in enumerate(RECT_OBSTACLES):
        if in_rect(point, obstacle):
            return f"inside rectangle #{i}"
    return None


def random_valid_point() -> Point:
    while True:
        point = (random.uniform(0, 1), random.uniform(0, 1))
        if blocking_obstacle(point) is None:
            return point


# --- evolution operators ---


def mut_fn(parent) -> List[Point]:
    x, y = parent.get_data()
    candidates = []
    for _ in range(MUTATIONS_PER_PARENT):
        nx = float(np.clip(x + random.gauss(0, MUTATION_SIGMA), 0.0, 1.0))
        ny = float(np.clip(y + random.gauss(0, MUTATION_SIGMA), 0.0, 1.0))
        candidates.append((nx, ny))
    return candidates


def crossover_fn(parent_1, parent_2) -> List[Point]:
    x1, y1 = parent_1.get_data()
    x2, y2 = parent_2.get_data()
    mx = (x1 + x2) / 2 + random.gauss(0, CROSSOVER_JITTER)
    my = (y1 + y2) / 2 + random.gauss(0, CROSSOVER_JITTER)
    return [(float(np.clip(mx, 0.0, 1.0)), float(np.clip(my, 0.0, 1.0)))]


def filter_fn(candidates) -> List[dict]:
    judgements = []
    for c in candidates:
        point = c.get_data()
        reason = blocking_obstacle(point)
        judgements.append({"accepted": reason is None, "feedback": reason or ""})
    return judgements


def embedding_fn(data: List[Point]) -> List[np.ndarray]:
    return [np.array(point) for point in data]


# --- plotting ---


def draw_obstacles(ax) -> None:
    for obstacle in CIRCLE_OBSTACLES:
        ax.add_patch(Circle(obstacle["center"], obstacle["radius"], color="0.3", alpha=0.4))
    for obstacle in RECT_OBSTACLES:
        width = obstacle["x_max"] - obstacle["x_min"]
        height = obstacle["y_max"] - obstacle["y_min"]
        ax.add_patch(
            Rectangle((obstacle["x_min"], obstacle["y_min"]), width, height, color="0.3", alpha=0.4)
        )


def plot_generation(generation, population: Population, result: StepResult)-> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    draw_obstacles(ax)

    archive_points = np.array([s.get_data() for s in population.get_archive()])
    if archive_points.size:
        ax.scatter(archive_points[:, 0], archive_points[:, 1], s=10, c="lightblue", alpha=0.5, label="archive")

    active_points = np.array([s.get_data() for s in population.get_active()])
    if active_points.size:
        ax.scatter(active_points[:, 0], active_points[:, 1], s=30, c="crimson", label="active")

    if generation is None:
        title = f"Seeds | archive={len(population.get_archive())}"
    else:
        title = (
            f"Generation {generation} | accepted={result.num_accepted} "
            f"selected={result.num_selected} archive={len(population.get_archive())}"
        )
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    filename = "gen_seed.png" if generation is None else f"gen_{generation:03d}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=120)
    plt.close(fig)


# --- main ---


def main():
    seeds = [random_valid_point() for _ in range(NUM_SEEDS)]

    population = Population()
    population.set_seeds(seeds)
    population.set_embedding_column(embedding_fn)

    engine = EvolutionEngine(population, selection_size=SELECTION_SIZE, log_path="examples/2d_example/logs")
    engine.set_mutation(mut_fn, num_mutation_samples=NUM_MUTATION_SAMPLES)
    engine.set_crossover(crossover_fn, num_crossover_samples=NUM_CROSSOVER_SAMPLES)
    engine.set_filters([filter_fn])
    engine.set_selection_config(SELECTION_CONFIG)


    for _ in range(NUM_GENERATIONS):
        result = engine.step()
        plot_generation(result.generation, population, result)

    print(f"Final archive size: {len(population.get_archive())}")
    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
