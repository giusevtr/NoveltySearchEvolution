"""Matplotlib helpers shared by the example's plotting scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DPI = 120


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_difficulty_histogram(
    counts: np.ndarray, title: str, output_path: Path, target: int | None = None
) -> None:
    """Bar chart of per-difficulty sample counts, with an optional target line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(counts)), counts)
    if target is not None:
        ax.axhline(target, color="red", linestyle="--", linewidth=1, label=f"target ({target})")
        ax.legend()
    ax.set_xlabel("difficulty (# of K student successes)")
    ax.set_ylabel("sample count")
    ax.set_title(title)
    save_figure(fig, output_path)
