"""Weights & Biases logging for EvolutionEngine.step()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import wandb

from .metrics import average_pairwise_distance, vendi_score
from .population import Population

if TYPE_CHECKING:
    from .evolution_engine import StepResult


def init_wandb_run(project: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    return wandb.init(project=project, name=run_name, config=config or {})


def log_step_metrics(wandb_run: Any, population: Population, step_result: "StepResult") -> None:
    """Log population composition and accepted-population diversity metrics for one generation.

    Called from EvolutionEngine.step(), right after a step completes.
    """
    accepted = population.get_accepted()
    metrics: Dict[str, Any] = {
        "population/accepted": len(accepted),
        "population/rejected": len(population.get_rejected()),
        "population/active": len(population.get_active()),
        "population/inactive": len(population.get_inactive()),
        "population/stale": len(population.get_stale()),
        "population/archive_size": len(population.get_archive()),
    }
    if len(accepted) >= 2:
        embeddings = population.get_embeddings(accepted)
        metrics["diversity/avg_pairwise_distance"] = average_pairwise_distance(embeddings)
        metrics["diversity/vendi_score"] = vendi_score(embeddings)
    wandb_run.log(metrics, step=step_result.generation)
