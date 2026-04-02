The file exists on my side. Here’s a fresh link:

# VibeRank

## Table of Contents
- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Running notebooks](#running-notebooks)
- [Dependency management](#dependency-management)
- [Data policy](#data-policy)
- [Results policy](#results-policy)
- [Development guidelines](#development-guidelines)
- [Typical workflow](#typical-workflow)
- [Config organization](#config-organization)
- [Configuration files](#configuration-files)
- [Testing](#testing)
- [Current status](#current-status)

VibeRank is a research codebase for running LLM-based ranking and comparison experiments across multiple datasets, models, and experimental pipelines.

Right now, the project is organized around:

- **Experiments**: `pairwise_comparisons`, `rank_centrality`
- **Datasets**: `mimic`, `hmls`
- **Models**: `deepseek`, `llama`

The repository is designed so that:

- **private/proprietary data stays local**
- **results can be version-controlled and shared**
- **code is importable cleanly from notebooks**
- **dependencies stay synchronized across collaborators**

---
---

## Repository structure

```text
.
├── README.md
├── .gitignore
├── pyproject.toml
├── notebooks/
│   ├── exploratory/
│   ├── debugging/
│   ├── pairwise_comparisons/
│   └── rank_centrality/
├── configs/
│   ├── experiments/
│   ├── datasets/
│   ├── models/
│   └── defaults.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── results/
│   ├── pairwise_comparisons/
│   ├── rank_centrality/
│   └── summaries/
├── scripts/
├── src/
│   └── viberank/
└── tests/
````

### Directory purposes

* `src/viberank/`
  Main Python package. Reusable logic should live here.

* `notebooks/`
  Jupyter notebooks for experiments, exploration, debugging, and analysis.

* `configs/`
  YAML configuration files for experiments, datasets, and models.

* `data/`
  Local-only data directory. This should never be committed to GitHub.

* `results/`
  Outputs, metrics, plots, and logs. These may be committed if they do not contain sensitive data.

* `scripts/`
  Reusable entry-point scripts for preparing data, running experiments, evaluation, and summarization.

* `tests/`
  Unit tests and sanity checks.

---

---

## Environment setup

We use a project-local Python environment and install the package so imports work cleanly from anywhere in the repository, including inside notebooks.

### Recommended setup with `uv`

From the root of the repository:

```bash
uv venv
source .venv/bin/activate
uv sync
```

This will:

* create a local virtual environment in `.venv/`
* install all dependencies from `pyproject.toml`
* install the `viberank` package into the environment

### Register the notebook kernel

After activating the environment, run:

```bash
python -m ipykernel install --user --name viberank --display-name "VibeRank"
```

Then, in Jupyter, select the kernel:

```text
VibeRank
```

---

---

## Running notebooks

Most experiments in this project are run from notebooks inside `notebooks/`.

Because the package is installed into the environment, imports should work directly:

```python
from viberank.datasets.loaders import load_dataset
from viberank.models.registry import get_model
from viberank.experiments.pairwise_comparisons import run_pairwise_experiment
```

You should not need to manually modify `sys.path`.

---

---

## Dependency management

This project uses `pyproject.toml` as the source of truth for dependencies.

### Installing dependencies

If you just pulled changes from GitHub and want to update your environment:

```bash
source .venv/bin/activate
uv sync
```

### Adding a new dependency

If you add a library, use:

```bash
uv add <package-name>
```

For example:

```bash
uv add scikit-learn
```

This updates:

* `pyproject.toml`
* `uv.lock`

After adding a dependency, commit both files so collaborators can sync cleanly.

### Collaborator workflow

If one collaborator adds a new package:

1. They run `uv add <package-name>`
2. They commit the updated `pyproject.toml` and `uv.lock`
3. The other collaborator pulls the changes
4. The other collaborator runs:

```bash
uv sync
```

This keeps environments aligned with minimal manual work.

---
---

## Data policy

The `data/` directory is for **local-only data**.

This includes:

* raw source data
* interim cleaned data
* processed data derived from proprietary inputs

These files should **not** be committed to GitHub.

Expected local structure:

```text
data/
├── raw/
│   ├── mimic/
│   └── hmls/
├── interim/
│   ├── mimic/
│   └── hmls/
└── processed/
    ├── pairwise_comparisons/
    └── rank_centrality/
```

---

## Results policy

The `results/` directory is intended for non-sensitive outputs that can be tracked in version control.

Examples:

* plots
* aggregated metrics
* evaluation summaries
* logs that do not contain proprietary data
* rankings or outputs safe for sharing

Typical structure:

```text
results/
├── pairwise_comparisons/
│   ├── mimic/
│   │   ├── deepseek/
│   │   └── llama/
│   └── hmls/
│       ├── deepseek/
│       └── llama/
├── rank_centrality/
│   ├── mimic/
│   │   ├── deepseek/
│   │   └── llama/
│   └── hmls/
│       ├── deepseek/
│       └── llama/
└── summaries/
```

---
---


## Development guidelines

A few conventions for keeping the codebase manageable:

### Put reusable logic in `src/viberank/`

Notebooks should be used for:

* orchestration
* exploration
* plotting
* debugging
* paper figure generation

Core logic should live in Python modules under `src/viberank/`.

### Keep notebooks thin

Try to avoid copying large blocks of logic into notebooks. Prefer calling functions from the package.

### Keep results reproducible

Whenever possible, save:

* metrics
* figures
* logs
* relevant config used for the run

alongside outputs in `results/`.

---
---

## Typical workflow

### First-time setup

```bash
git clone <repo-url>
cd VibeRank
uv venv
source .venv/bin/activate
uv sync
python -m ipykernel install --user --name viberank --display-name "VibeRank"
```

### Starting work later

```bash
cd VibeRank
source .venv/bin/activate
jupyter notebook
```

### After pulling collaborator changes

```bash
git pull
source .venv/bin/activate
uv sync
```

---
---

## Config organization

Configurations are split by concern:

* `configs/experiments/` for experiment-specific settings
* `configs/datasets/` for dataset-specific settings
* `configs/models/` for model-specific settings

This allows runs to be composed cleanly from independent pieces.

Examples:

* `pairwise_comparisons + mimic + llama`
* `rank_centrality + hmls + deepseek`

## Configuration files

The `configs/` directory stores YAML files that define the pieces of an experimental run. Splitting configuration by concern makes runs easier to understand, reuse, and reproduce.

```text
configs/
├── experiments/
│   ├── pairwise_comparisons.yaml
│   └── rank_centrality.yaml
├── datasets/
│   ├── mimic.yaml
│   └── hmls.yaml
├── models/
│   ├── deepseek.yaml
│   └── llama.yaml
└── defaults.yaml
```

### Why split configs into multiple files?

Each run in VibeRank is defined by three main axes:

* **experiment**
* **dataset**
* **model**

These vary independently. For example, you may want to run:

* `pairwise_comparisons + mimic + llama`
* `pairwise_comparisons + hmls + deepseek`
* `rank_centrality + mimic + deepseek`

Rather than hardcoding every combination in Python, the repository keeps each piece in its own config file and combines them at runtime.

This has several benefits:

* avoids duplicating code for every experiment/dataset/model combination
* makes runs easier to reproduce
* keeps experiment settings organized and readable
* allows collaborators to inspect and modify one part of a run without touching everything else

---

### `configs/experiments/`

These files define **what the experiment is doing**.

Examples of settings that belong here:

* experiment name
* experiment type
* number of runs
* sampling settings
* prompt template names
* aggregation method
* whether to save intermediate outputs

For example:

* `pairwise_comparisons.yaml` might specify how pairwise judgments are generated
* `rank_centrality.yaml` might specify how pairwise outputs are aggregated into a ranking

In other words, experiment configs control the **pipeline logic**.

---

### `configs/datasets/`

These files define **how a dataset should be loaded and interpreted**.

Examples of settings that belong here:

* dataset name
* local data path
* split names
* column mappings
* preprocessing flags
* token or text length limits
* filtering rules

For example:

* `mimic.yaml` may point to local MIMIC data and define dataset-specific preprocessing
* `hmls.yaml` may define the path and formatting rules for homelessness-service data

Dataset configs control the **input side** of a run.

---

### `configs/models/`

These files define **which model is being used and how it should be called**.

Examples of settings that belong here:

* model name
* provider or backend
* inference parameters
* temperature
* max tokens
* batch size
* retry behavior
* local vs remote execution flags

For example:

* `deepseek.yaml` may contain settings specific to the DeepSeek backend
* `llama.yaml` may contain settings for a local or hosted LLaMA model

Model configs control the **inference behavior** of a run.

---

### `configs/defaults.yaml`

This file is for settings that should be shared across many runs.

Examples:

* default random seed
* default output directory conventions
* common logging settings
* default save options
* global plotting or formatting settings

This avoids repeating the same values across multiple experiment, dataset, or model files.

---

### How configs are used together

A single run is typically formed by combining:

* one experiment config
* one dataset config
* one model config
* optionally shared defaults

Conceptually, a run looks like:

```text
experiment + dataset + model
```

For example:

```text
pairwise_comparisons + mimic + llama
```

The code can then load the corresponding YAML files and compose them into one runtime configuration.

This keeps the code modular:

* Python modules define **how** things run
* YAML configs define **what** should be run

---

### Practical guideline

As a rule of thumb:

* put **pipeline logic settings** in `configs/experiments/`
* put **data-specific settings** in `configs/datasets/`
* put **model/inference settings** in `configs/models/`
* put **shared fallback settings** in `configs/defaults.yaml`

If a setting would change when you switch from `mimic` to `hmls`, it probably belongs in a dataset config.
If it would change when you switch from `llama` to `deepseek`, it probably belongs in a model config.
If it would change when you switch from `pairwise_comparisons` to `rank_centrality`, it probably belongs in an experiment config.

---

### Reproducibility

One benefit of this structure is that results can be traced back to the exact configuration used to generate them.

Whenever possible, save the effective config used for a run alongside the outputs in `results/`. This makes it easier to:

* rerun old experiments
* compare settings across runs
* debug unexpected differences
* share experimental details with collaborators

---

---

## Testing

Run tests from the repository root:

```bash
pytest
```

As the codebase grows, tests should cover:

* dataset loading
* model wrappers
* pairwise comparison pipelines
* rank aggregation logic
* metrics and summary functions

---

---

## Current status

Initial scaffold for:

* multiple datasets
* multiple models
* multiple experiment types
* notebook-driven workflows
* local-only private data
* shared dependency management

---


