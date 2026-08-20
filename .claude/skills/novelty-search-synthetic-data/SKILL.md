---
name: novelty-search-synthetic-data
description: Build a novelty-search application on top of this repo's novelty_search_evolution/ framework (EvolutionEngine + Population) — generate a diverse synthetic dataset or explore a space by mutating/recombining seed examples, filtering invalid ones, and keeping the most novel survivors. Use when asked to write a mut_fn/crossover_fn/filter_fn/embedding_fn, tune nn-k/distance/archive_update_prob, or wire an LLM into novelty search.
---

# Novelty search synthetic data

Recipe for implementing a novelty-search application with this repo's
domain-agnostic `novelty_search_evolution/` framework. Read `README.md` first
for the full `EvoSample`/`Population`/`EvolutionEngine` API — this skill is
the "how do I actually fill in the four functions and pick config" cheat
sheet, not a restatement of the API.

## The four functions you write

All four operate on plain `data` payloads (whatever your candidates are —
numbers, dicts, strings) via `EvoSample.get_data()`. Never touch `EvoSample`
lifecycle/genealogy yourself; the engine does that.

- **`mut_fn(parent) -> [data, ...]`** — perturb one parent, return 1+ new
  candidates. Pure-Python: e.g. Gaussian jitter (`examples/2d_example/run.py`).
  LLM-backed: a closure capturing your client, e.g.
  `make_mut_fn(teacher, student, ...)` in
  `examples/gsm8k_distillation/scripts/run_novelty_search_augmentatin.py` —
  batch all candidates for the generation into as few LLM calls as possible
  rather than one call per candidate.
- **`crossover_fn(parent_1, parent_2) -> [data, ...]`** — recombine two
  parents (e.g. midpoint + jitter, or an LLM blending two "techniques").
- **`filter_fn(candidates) -> [{"accepted": bool, "feedback": str}, ...]`**
  — one judgement per candidate, same order. `feedback` is free text; it can
  double as metadata even in an always-accept filter (e.g.
  `feedback="class=3"` when the filter is really doing labeling, not
  rejecting).
- **`embedding_fn(data_list) -> [np.ndarray, ...]`** — called **batched**;
  batch/parallelize expensive (LLM) embedding calls rather than looping
  per-sample (`ThreadPoolExecutor`, as in `examples/gsm8k_distillation`'s
  embedding step). Can be the identity function if `data` already is a
  numeric vector (`examples/2d_example`).

## Config cheat sheet

| Parameter | Guidance |
| --- | --- |
| `distance` | `"euclidean"` for raw numeric/geometric data; `"cosine"` for semantic/text embeddings. |
| `nn-k` | Scales with archive size: ~3-5 for small/simple spaces, 10-50 for large semantic archives. |
| `archive_update_prob` | Trade-off between archive churn and slot preciousness: lower (~0.05) when the space is large/expensive to explore and archive slots shouldn't be wasted; higher (~0.3) when discovery is cheap. |
| `num_mutation_candidates` / `num_crossover_samples` | Set per-generation candidate volume. Parents/pairs are sampled *with replacement*, so either can exceed the active population size. |
| `parallel` (on `set_mutation`/`set_crossover`) | Defaults to `True` — runs `mut_fn`/`crossover_fn` calls concurrently via `ThreadPoolExecutor`. Good default for I/O-bound (LLM) calls; set `False` for cheap synchronous functions where thread overhead isn't worth it. |

## Two stopping-condition patterns

- **Fixed generations**: `engine.run(n)` or a `for` loop — simplest, good
  default (`examples/2d_example`).
- **Target dataset size**: a `while` loop checking
  `len(population.get_accepted())` against a target, with a generation-count
  safety cap — use when you want *N* accepted samples rather than *N*
  generations (`examples/gsm8k_distillation`).

## Worked examples

- **`examples/2d_example/run.py`** — simplest full runnable reference (pure
  numpy, no LLM). Run it:
  ```bash
  pip install -r examples/2d_example/requirements.txt
  PYTHONPATH=. python examples/2d_example/run.py
  ```
  `mut_fn` jitters+clips a point, `crossover_fn` jitters the midpoint,
  `filter_fn` rejects points inside obstacles, `embedding_fn` is the
  identity. Config: `nn-k=5`, `distance="euclidean"`,
  `archive_update_prob=0.3`.
- **`examples/gsm8k_distillation/scripts/run_novelty_search_augmentatin.py`**
  — LLM-backed reference (teacher/student distillation). `mut_fn`/`crossover_fn`
  are factories capturing LLM clients and batch all calls for a generation
  together; `filter_fn` validates output format; `embedding_fn` batches
  embedding calls and concatenates normalized sub-vectors so different
  aspects of the data (semantics, metadata) contribute equally to novelty
  distance. Config: `nn-k=10`, `distance="cosine"`,
  `archive_update_prob=0.3`. Also shows a `--smoke-test` cheap-config flag —
  write one for any LLM-backed novelty search so you can dry-run the wiring
  before a full (costly) run — and optional `engine.set_wandb_logging(...)`
  for run tracking.
