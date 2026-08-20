# NoveltySearchEvolution

A novelty-search-driven evolutionary framework (`novelty_search_evolution/`): generate candidates
via mutation and crossover, filter them, score them by novelty against an archive, and select a
diverse, evolving population — domain-agnostic, driven entirely by functions you supply.

The framework is designed with LLM-backed use in mind: `mut_fn`, `crossover_fn`, `filter_fn`, and
`embedding_fn` are typically implemented by prompting an LLM (to mutate/recombine candidates,
judge/filter them, or embed them for novelty scoring). No LLM client is bundled — bring whichever
one you like (the OpenAI SDK, AWS Bedrock's `boto3` client, LangChain, etc.) and call it from
inside your own functions.

## Installation

```bash
cd ~/<path>/NoveltySearchEvolution
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
pip install -e .
```

This installs the core novelty search framework (`novelty_search_evolution/`).

To run the runnable demo, install its extra dependencies separately — see
[Example: 2D novelty search](#example-2d-novelty-search) below.

## Core framework: Novelty Search Evolution

`novelty_search_evolution/` implements the evolution loop:

- **`EvoSample`** — one candidate: its `data`, lifecycle `status`
  (`STALE → ACTIVE ⇄ INACTIVE`, or terminal `REJECTED`), and genealogy (`parents`/`children`).
- **`Population`** — owns every `EvoSample`, the novelty **archive** (the reference set novelty
  is scored against), and batched embedding computation via a pluggable `embedding_fn`. Status
  accessors (`get_active`, `get_inactive`, `get_stale`, `get_rejected`) filter by lifecycle
  status; `get_accepted()` returns everything except `REJECTED` — the set you typically want for
  a final/checkpointed dataset.
- **`EvolutionEngine`** — drives `step()`: select parents → generate candidates via your
  `mut_fn`/`crossover_fn` → run them through your `filter_fn`(s) → embed the survivors → select
  the most novel via `novelty_selection.score()` / `DefaultSelectionEngine` (k-NN novelty,
  greedy diverse pick) → update the archive.

Minimal usage:

```python
from novelty_search_evolution import EvolutionEngine, Population

def mut_fn(parent):
    ...          # -> list of new candidate `data` values

def crossover_fn(parent_1, parent_2):
    ...          # -> list of new candidate `data` values

def filter_fn(candidates):
    ...          # -> [{"accepted": bool, "feedback": str}, ...] per candidate

def embedding_fn(data):
    ...          # -> list of np.ndarray, one per input, batched

population = Population()
population.set_seeds([...])              # initial data values
population.set_embedding_column(embedding_fn)

engine = EvolutionEngine(population, selection_size=30)
engine.set_mutation(mut_fn, num_mutation_candidates=15)
engine.set_crossover(crossover_fn, num_crossover_samples=5)
engine.set_filters([filter_fn])
engine.set_selection_config({
    "nn-k": 5,
    "distance": "euclidean",       # or "cosine"
    "archive_update_prob": 0.3,
})

for _ in range(30):
    result = engine.step()         # or engine.run(30)
    print(result)                  # StepResult: counts + newly active/inactive/archived ids

accepted = population.get_accepted()   # every sample that passed filter_fn: stale, active, or
                                        # inactive — excludes only REJECTED samples
```

`num_mutation_candidates` is the target total number of mutation candidates generated per
generation (one `mut_fn` call per candidate, assuming `mut_fn` returns a single candidate per
call — if it returns more, the actual count is a multiple of this). To hit that target, parents
are sampled *with replacement* from the active population, so `num_mutation_candidates` can
exceed the number of currently active parents. `num_crossover_samples` is the number of parent
pairs drawn per generation; each pair is drawn independently, so parents are already effectively
sampled with replacement across the full set of pairs. By default (`parallel=True` on both
`set_mutation` and `set_crossover`), the per-candidate `mut_fn`/`crossover_fn` calls run
concurrently via a `ThreadPoolExecutor` — well suited to I/O-bound calls like LLM requests; pass
`parallel=False` to either call to run its operator sequentially instead.

## Logging: Weights & Biases

`EvolutionEngine` can optionally log each `step()` to [Weights & Biases](https://wandb.ai)
(`wandb`, a core dependency pulled in by `pip install -e .`).

Setup: you need a W&B account. Either run `wandb login` once, or set the `WANDB_API_KEY`
environment variable — `set_wandb_logging` just calls `wandb.init(...)` under the hood, so it
picks up the standard `wandb` env/config resolution; the framework does not manage the key
itself.

```python
engine.set_wandb_logging(project="novelty-search-demo", run_name="run-1")

for _ in range(30):
    result = engine.step()

engine.finish_wandb()
```

Once enabled, every `step()` automatically logs, no extra calls needed:

- population composition — `population/accepted`, `population/rejected`, `population/active`,
  `population/inactive`, `population/stale`, `population/archive_size`
- diversity metrics on the accepted population's embeddings (once it has 2+ members) —
  `diversity/avg_pairwise_distance`, `diversity/vendi_score`

### Diversity metrics

`novelty_search_evolution/metrics.py` provides diversity metrics that operate on any `(N, D)`
embedding array, independent of wandb:

- **`vendi_score(embeddings)`** — the effective number of distinct samples: `exp(entropy)` of the
  eigenvalues of the mean-normalized cosine-similarity kernel over the embeddings. It's a
  redundancy-aware diversity measure — a population of near-duplicates scores low even if the
  average pairwise distance looks fine, because near-duplicate embeddings contribute correlated
  (not independent) mass to the kernel's spectrum. Higher means more effectively-distinct
  samples.
- **`average_pairwise_distance(embeddings)`** — mean cosine distance over all pairs; a simpler,
  non-redundancy-aware diversity measure.
- **`mean_min_distance_to_reference(embeddings, reference_embeddings)`** — for each reference
  (e.g. ground-truth) embedding, the cosine distance to its nearest neighbor in `embeddings`,
  averaged over all reference embeddings; lower means better coverage of the reference
  distribution.

All three are importable directly from `novelty_search_evolution`.

## Skills

Since the framework is domain-agnostic, task-specific applications are packaged as Claude Code
skills under `.claude/skills/<skill-name>/SKILL.md` — one skill per use case, each documenting a
ready-to-use recipe (parameter guidance + runnable pattern) for that task:

- **`novelty-search-synthetic-data`** — recipe for implementing `mut_fn`/`crossover_fn`/
  `filter_fn`/`embedding_fn`, a config cheat-sheet (`nn-k`, `distance`, `archive_update_prob`,
  candidate volume), and worked-example pointers (`examples/2d_example` for a runnable
  pure-Python start, `examples/gsm8k_distillation` for the LLM-backed pattern).

## Example: 2D novelty search

`examples/2d_example/run.py` runs the framework on an easy-to-visualize problem: samples are
points in `[0,1]×[0,1]`, with 2 circular and 1 rectangular obstacle carved out of the grid.
Mutation perturbs points with Gaussian noise, crossover takes parent midpoints, and the filter
rejects anything landing inside an obstacle — novelty search then drives coverage of the
remaining free space.

```bash
pip install -r examples/2d_example/requirements.txt
PYTHONPATH=. python examples/2d_example/run.py
```

Each generation (plus an initial seed frame) is rendered to
`examples/2d_example/output/gen_*.png` — archive coverage in faint blue, the current active
frontier in red, obstacles as gray shapes — so you can watch coverage spread across the grid over
time. The `output/` directory is gitignored and regenerated on every run.

## Architecture

- **`novelty_search_evolution/`** — novelty search evolution framework
  - `sample.py` — `EvoSample`, `Status` lifecycle
  - `population.py` — `Population`: sample storage, archive, embeddings
  - `evolution_engine.py` — `EvolutionEngine`, `StepResult`: the `step()`/`run()` loop
  - `selection_engine.py` — `SelectionEngine` protocol, `DefaultSelectionEngine`
  - `novelty_selection.py` — k-NN novelty `score()`
  - `population_viewer.py` — renders the per-step population snapshot to HTML
  - `metrics.py` — diversity metrics (Vendi score, average pairwise distance, mean min distance
    to reference)
  - `wandb_logging.py` — optional W&B logging for `EvolutionEngine.step()`

- **`examples/2d_example/`** — runnable demo: 2D novelty search with obstacles, saves
  per-generation coverage plots

## Testing

Run the full test suite (no GPU/network required):

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_evolution_engine.py -v       # EvolutionEngine tests
pytest tests/test_population.py -v             # Population tests
pytest tests/test_sample.py -v                 # EvoSample tests
pytest tests/test_novelty_selection.py -v      # Novelty scoring tests
pytest tests/test_population_viewer.py -v      # Population viewer tests
pytest tests/test_selection_engine.py -v       # DefaultSelectionEngine tests
pytest tests/examples -v                       # Example (2D demo, GSM8K helpers) tests
```

`tests/examples/` covers the pure logic of the examples; tests needing an optional dependency
(e.g. matplotlib for the 2D demo) are skipped when it isn't installed.

Coverage report (`pytest-cov` comes with the `dev` extra):

```bash
pytest tests/ --cov=novelty_search_evolution --cov=examples --cov-report=term-missing
```
