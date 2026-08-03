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
  is scored against), and batched embedding computation via a pluggable `embedding_fn`.
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
engine.set_mutation(mut_fn, num_mutation_samples=15)
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
```

## Skills

Since the framework is domain-agnostic, task-specific applications are packaged as Claude Code
skills under `.claude/skills/<skill-name>/SKILL.md` — one skill per use case, each documenting a
ready-to-use recipe (parameter guidance + runnable pattern) for that task:

- **`novelty-search-synthetic-data`** — generate a diverse synthetic dataset by mutating and
  recombining seed examples, filtering invalid ones, and keeping the most novel survivors;
  explains how each novelty-search parameter (`nn-k`, `distance`, `top_k`,
  `archive_update_prob`, mutation/crossover volume, `embedding_fn`) shapes the output, and how
  to wire in an LLM client of your choice (e.g. LangChain, or a provider SDK directly) for
  LLM-backed mutation/filtering/embedding.

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
```
