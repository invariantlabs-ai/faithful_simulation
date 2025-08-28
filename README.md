# Faithful Simulation of User–Agent–Environment Interactions for Scalable LLM Agent Evaluation

<!-- [![Paper](https://img.shields.io/badge/arXiv-2025XXXX-blue)](https://arxiv.org/abs/XXXX.XXXXX) -->

## Overview

This repository contains the source code accompanying the paper:

> **Faithful Simulation of User–Agent–Environment Interactions for Scalable LLM Agent Evaluation**
> Aleksei Kudrinskii\*, Saibo Geng, Marc Fischer, Luca Beurer-Kellner
> *École Polytechnique Fédérale de Lausanne (EPFL) & Snyk*

Large Language Models (LLMs) are evolving from simple chatbots into **interactive agents**. Evaluating such systems requires realistic environments and diverse user interactions—but existing benchmarks are limited, costly, or fail to capture real-world complexity.

Our framework provides a **faithful and scalable simulation pipeline** for evaluating tool-using LLM agents. It automatically generates tasks, simulates closed-loop conversations with user and environment archetypes, and evaluates performance using principled metrics.

---

## Key Features

* **Automated Task Generation**

  * Build tasks from a **Tool–Relationship Graph (TRG)** derived from Model Context Protocol (MCP) tool specs.
* **Closed-Loop Simulation**

  * Three actors: **Agent (LLM)**, **User Archetype**, **Environment Archetype**.
  * Configurable archetypes: e.g., *Planner, Improviser, Goal-Shifter* for users; *Perfect, Buggy, Adversarial* for environments.
* **Evaluation Metrics**

  * *Trace Alignment*: procedural fidelity to ground-truth tool paths.
  * *Goal Achievement*: end-to-end task success judged by independent LLM.
  * *Simulation Faithfulness*: audits ensuring realistic behavior of users and environments.

---

## Installation

```bash
git clone https://github.com/akudrinsky/afma.git
cd afma
uv sync
source .venv/bin/activate
pip install -e .
```

---

## Usage Example

Run a simulation with a chosen agent, user archetype, and environment:

```bash
afma run-pipeline configs/big_eval/config_gpt4-1.yaml -o ./results/gpt-4-1/big_eval/
```

This will:

1. Generate a multi-step task from the TRG.
2. Run a conversation between User–Agent–Environment.
3. Save logs and evaluation metrics in `./results/`.

---

---
## Results viewing

Quick ways to inspect and visualize results saved under `./results/`.

- Conversation Viewer (Streamlit): interactive browser to filter, read chats, inspect toolcalls, and view trace alignments.
  - Why: explore failures/successes, compare personalities, and drill into conversations.
  - Run:
    ```bash
    uv run streamlit run tools/conversation_viewer.py
    ```

- Personality Radar Chart: plot average Trace Alignment and Goal Achievement by user and environment personalities across one or more models.
  - Why: see how archetypes affect performance and compare models at a glance.
  - Run (pass model result roots or a parent `results/` dir):
    ```bash
    uv run tools/personality_radar_chart.py ./results/gpt-4-1 ./results/gpt-4-1-mini
    ```

- Personality Radar Chart by Length: same as above, broken down by task (tool-path) length.
  - Why: understand how complexity impacts models for different personalities.
  - Run:
    ```bash
    uv run tools/personality_radar_chart_by_length.py ./results
    ```

- Generate Personality Analysis (tables in console): prints user×environment pivots and sequence-length breakdowns.
  - Why: quick, reproducible tabular summaries for papers/notes.
  - Run (point to a specific run folder like `.../big_eval`):
    ```bash
    uv run tools/generate_personality_analysis.py ./results/gpt-4-1/big_eval
    ```

- Extract Trace Scores (LaTeX): emits LaTeX tables for goals/alignment by archetype and by sequence length, plus inner-trace-set variance.
  - Why: drop-in tables for manuscripts; quantify variability across instantiations.
  - Run:
    ```bash
    uv run tools/extract_trace_scores.py ./results/gpt-4-1/big_eval --output-file results/aggregated_performance/tables.tex
    ```

- Aggregate Performance by Sequence Length (multi-run): compares models on Goal Achievement vs ground-truth path length; saves chart and LaTeX.
  - Why: compare scaling with task complexity across models.
  - Run (example with defaults or pass explicit runs):
    ```bash
    uv run tools/aggregate_performance_metrics.py --runs \
      gemini:$(pwd)/results/gemini/0806_user_goal_env_precond_validity_gpt4.1_assessor \
      o4-mini:$(pwd)/results/o4-mini/0806_user_goal_env_precond_validity_gpt4.1_assessor
    ```

- Trace–Goal Correlation Analysis (LaTeX + console): measures correlation between trace fidelity and end-to-end success per model/personality.
  - Why: test whether procedural faithfulness mediates goal achievement or if safe alternatives exist.
  - Run:
    ```bash
    uv run tools/trace_goal_correlation_analysis.py ./results -o results/aggregated_performance/trace_goal_correlation --save-data
    ```

- Scores Visualization (demo charts): creates circular bar plots from example aggregated numbers.
  - Why: quick illustrative figures; replace with your own aggregates if desired.
  - Run:
    ```bash
    uv run tools/scores_visualization.py
    ```

## Example Results

* Environment reliability is the **dominant factor** in agent success.
* User interaction styles strongly modulate outcomes.
* Strict trace fidelity correlates with—but does not fully determine—end-to-end goal achievement.

---

## Repository Structure

High-level layout. `src/afma` is the main framework; `afma_testing` hosts tests that audit the faithfulness of the simulated User/Environment.

```text
.
├── src/afma/                 # Main framework (installed package `afma`)
│   ├── mcp_parser/           # Utilities to parse MCP servers' configurations, adapted from https://github.com/invariantlabs-ai/mcp-scan
│   ├── simulation/           # Closed-loop User–Agent–Environment runtime
│   ├── evaluation/           # Trace alignment, goal achievement, faithfulness
│   ├── alignment_visualization.py  # Graphs used by the viewer
│   └── ...                   # Other framework modules
│
├── afma_testing/             # Faithfulness tests for simulation components
│   ├── environment/          # Validity of environment state, transitions, comparators
│   ├── user_generation/      # User goal clarity, environment completeness checks
│   ├── personality_adherence/# Do simulated users follow their archetype traits?
│   ├── user_goal_adherence/  # Does the agent adhere to the user goal under dynamics?
│   └── severity_assessment/  # Manual/LLM labeling utilities for error severity
│
├── configs/                  # Experiment configs
│   ├── big_eval/             # Main evaluation config suites (YAML)
│   └── user_goal_env_validity/ # Validity audit configs (F1) (YAML)
├── agent_mcp_configs/        # MCP tool specs used to build the TRG
├── litellm_configs/          # Model routing/providers for LiteLLM
├── tools/                    # Result viewers, charts, correlation + LaTeX exporters
├── results/                  # Simulation outputs per run (JSON logs, summaries)
├── main.py                   # CLI entry to run pipelines
├── config.yaml               # Default run configuration
├── pyproject.toml            # Project metadata and dependencies
└── README.md
```

---

## Citation

If you use this framework in your research, please cite:
hope to be citable soon
