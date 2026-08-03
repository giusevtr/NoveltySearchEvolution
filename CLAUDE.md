# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
pip install -e .
```

Run the full test suite:

```bash
pytest tests/ -v
```

Run a single test module:

```bash
pytest tests/test_evolution_engine.py -v             # single file
pytest tests/test_population.py -v
```

Run the 2D novelty search example (self-contained, has its own requirements file):

```bash
pip install -r examples/2d_example/requirements.txt
PYTHONPATH=. python examples/2d_example/run.py
```
Outputs per-generation coverage plots to `examples/2d_example/output/` (gitignored).

## Architecture

This repo is a novelty-search-driven evolution framework, `novelty_search_evolution/`. See
`README.md` for a fuller walkthrough.

### `novelty_search_evolution` — the evolution loop

Everything is domain-agnostic and driven entirely by four user-supplied functions:
`mut_fn`, `crossover_fn`, `filter_fn`, `embedding_fn`. `examples/2d_example/run.py` is a working
reference implementation of all four for a concrete (2D point / obstacle) domain. These functions
are typically LLM-backed — the framework doesn't bundle an LLM client, so bring whichever one you
like (OpenAI SDK, AWS Bedrock, LangChain, etc.) and call it from inside your own functions.

- **`EvoSample`** (`sample.py`) — one candidate: its `data`, lifecycle `status`, and genealogy
  (`parents`/`children`). Status transitions **only** happen through named setters
  (`set_active`/`set_stale`/`set_inactive`/`set_reject`) — never by assigning `_status`
  directly, even from within `Population`/`EvolutionEngine`. Lifecycle:
  `STALE → ACTIVE ⇄ INACTIVE`, or terminal `REJECTED`.
- **`Population`** (`population.py`) — owns every `EvoSample` plus the novelty **archive** (the
  reference set new candidates are scored against), and does batched embedding computation via
  the injected `embedding_fn` (skips samples that already have an embedding).
- **`EvolutionEngine`** (`evolution_engine.py`) — `step()` runs one generation: select parents →
  generate candidates via `mut_fn`/`crossover_fn` → run them through `filter_fn`(s) (a filter
  can short-circuit remaining filters by rejecting a candidate) → embed survivors → novelty-select
  via `DefaultSelectionEngine` → update the archive. Returns a `StepResult` with counts and
  newly active/inactive/archived ids. `run(n)` just calls `step()` n times.
- **`novelty_selection.score()`** / **`selection_engine.py`** — k-NN novelty scoring
  (`scipy.spatial.distance.cdist`) and `DefaultSelectionEngine`'s greedy sequential selection
  (each pick is scored against the archive *and* prior picks in the same round, to avoid
  near-duplicate selections within one generation).
- **`population_viewer.py`** — renders the per-step population snapshot to HTML.
