#!/usr/bin/env python3
"""
Aggregate Performance Metrics (Trace Alignment & Goal Achievement) by Sequence Length across models.

This script plots average trace alignment and goal achievement scores to show overall
performance trends across different task complexities.

Usage examples:
  uv run tools/aggregate_performance_metrics.py \
    --runs \
      gemini:/Users/kudrinsky/Documents/afma/results/gemini/0806_user_goal_env_precond_validity_gpt4.1_assessor \
      o4-mini:/Users/kudrinsky/Documents/afma/results/o4-mini/0806_user_goal_env_precond_validity_gpt4.1_assessor \
      gpt-4-1-mini:/Users/kudrinsky/Documents/afma/results/gpt-4-1-mini/0806_user_goal_env_precond_validity_gpt4.1_assessor \
      gpt-4-1:/Users/kudrinsky/Documents/afma/results/gpt-4-1/0806_user_goal_env_precond_validity_gpt4.1_assessor
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is importable so we can import from `tools`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import analysis helpers from existing tool
from tools.extract_trace_scores import (  # type: ignore
    load_experiment_data,
    get_similarity_score,
    get_goal_achievement_score,
    get_all_sequence_lengths,
)


def parse_runs_arg(runs: List[str]) -> Dict[str, Path]:
    """Parse --runs entries of the form name:/abs/path into a dict."""
    result: Dict[str, Path] = {}
    for entry in runs:
        if ":" not in entry:
            raise ValueError(f"Invalid --runs entry (expected name:/abs/path): {entry}")
        name, path_str = entry.split(":", 1)
        p = Path(path_str).expanduser().resolve()
        result[name] = p
    return result


def default_runs(repo_root: Path) -> Dict[str, Path]:
    base = repo_root / "results"
    return {
        "gemini": base / "gemini" / "0806_user_goal_env_precond_validity_gpt4.1_assessor",
        "o4-mini": base / "o4-mini" / "0806_user_goal_env_precond_validity_gpt4.1_assessor",
        "gpt-4-1-mini": base / "gpt-4-1-mini" / "0806_user_goal_env_precond_validity_gpt4.1_assessor",
        "gpt-4-1": base / "gpt-4-1" / "0806_user_goal_env_precond_validity_gpt4.1_assessor",
        "gpt-5": base / "gpt-5" / "0806_user_goal_env_precond_validity_gpt4.1_assessor",
        "gpt-5-mini": base / "gpt-5-mini" / "0806_user_goal_env_precond_validity_gpt4.1_assessor",
        "claude-sonnet-4": base / "claude-sonnet-4" / "0806_user_goal_env_precond_validity_gpt4.1_assessor",
    }


def analyze_performance_by_sequence_size(conversations: List[Dict], trace_alignments: Dict) -> Dict[str, Dict[str, float]]:
    """Analyze average trace alignment and goal achievement by GT sequence size."""
    size_groups = defaultdict(lambda: {'trace_scores': [], 'goal_scores': []})
    
    for conv in conversations:
        gt_size = len(conv.get('user_source', []))  # get_gt_sequence_size equivalent
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        size_groups[gt_size]['trace_scores'].append(trace_score)
        size_groups[gt_size]['goal_scores'].append(goal_score)
    
    # Calculate averages for each size
    results = {}
    for gt_size, scores in size_groups.items():
        trace_mean = statistics.mean(scores['trace_scores']) if scores['trace_scores'] else 0.0
        goal_mean = statistics.mean(scores['goal_scores']) if scores['goal_scores'] else 0.0
        
        results[f"Size {gt_size}"] = {
            'trace_alignment': trace_mean,
            'goal_achievement': goal_mean,
            'sample_count': len(scores['trace_scores'])
        }
    
    return results


def collect_performance_by_size(runs: Dict[str, Path]) -> Tuple[pd.DataFrame, List[int]]:
    """Return dataframe with columns: size(int), model(str), trace_alignment(float), goal_achievement(float)."""
    records = []
    all_sizes: set[int] = set()

    for model_name, run_dir in runs.items():
        conversations, trace_alignments, _summary = load_experiment_data(run_dir)
        size_analysis = analyze_performance_by_sequence_size(conversations, trace_alignments)

        # gather sizes present in conversations for consistent x-axis
        sizes_here = get_all_sequence_lengths(conversations)
        for s in sizes_here:
            all_sizes.add(s)

        # Rename gemini to gemini-2.5-flash
        display_name = "gemini-2.5-flash" if model_name == "gemini" else model_name

        for size_key, vals in size_analysis.items():
            try:
                size = int(size_key.split()[1])
            except Exception:
                continue
            records.append(
                {
                    "size": size,
                    "model": display_name,
                    "trace_alignment": float(vals.get("trace_alignment", 0.0)),
                    "goal_achievement": float(vals.get("goal_achievement", 0.0)),
                    "sample_count": int(vals.get("sample_count", 0)),
                }
            )

    df = pd.DataFrame.from_records(records)
    sizes_sorted = sorted(all_sizes)
    return df, sizes_sorted


def make_goal_achievement_chart(df: pd.DataFrame, sizes_sorted: List[int], out_path: Path) -> None:
    """Create a chart with goal achievement metric only."""
    # Clean, publication-ready styling
    plt.style.use("default")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pale, muted color palette
    colors = {
        "gemini-2.5-flash": "#7FB3D5",      # Pale blue
        "o4-mini": "#BB8FCE",               # Pale purple
        "gpt-4-1-mini": "#F8C471",          # Pale orange
        "gpt-4-1": "#F1948A",               # Pale red/coral
        "gpt-5": "#85C1E9",                 # Light blue
        "gpt-5-mini": "#D7BDE2",            # Light lavender
        "claude-sonnet-4": "#A9DFBF"        # Pale green
    }
    
    # Order models by goal achievement performance
    # This ensures better models appear on the right consistently
    model_performance = {}
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        avg_goal = model_data["goal_achievement"].mean()
        model_performance[model] = avg_goal
    
    # Sort by performance (worst to best, so best appears on right)
    model_order = sorted(model_performance.keys(), key=lambda x: model_performance[x])
    
    # Prepare data for grouped bar chart
    num_models = len(model_order)
    bar_width = 0.12 if num_models > 5 else 0.2  # Narrower bars for more models
    x_positions = range(len(sizes_sorted))
    
    # Plot Goal Achievement only
    for i, model_name in enumerate(model_order):
        model_data = df[df["model"] == model_name].copy()
        model_data = model_data.set_index("size").reindex(sizes_sorted)
        
        y_values = model_data["goal_achievement"].fillna(0).values
        x_offset = [x + i * bar_width for x in x_positions]
        
        bars = ax.bar(
            x_offset, 
            y_values, 
            bar_width, 
            label=model_name,
            color=colors.get(model_name, "#666666"),
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add rotated value labels on top of bars
        for bar, value in zip(bars, y_values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=7, 
                       color='#333333', rotation=45)
    
    # Styling
    ax.set_xlabel('GT Tool Path Length', fontsize=10, fontweight='medium')
    ax.set_ylabel('Goal Achievement', fontsize=10, fontweight='medium')
    
    # Set x-axis (center ticks between all bars)
    ax.set_xticks([x + bar_width * (num_models - 1) / 2 for x in x_positions])
    ax.set_xticklabels(sizes_sorted, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # Clean grid and spines
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Dynamically crop the minimum to improve visual separation of bars
    try:
        global_min = float(df["goal_achievement"].min())
        global_max = float(df["goal_achievement"].max())
        value_range = max(1e-6, global_max - global_min)
        padding = max(0.02, value_range * 0.1)
        lower_bound = max(0.0, global_min - padding)
        upper_bound = 1.0
        # Ensure at least a reasonable visible range
        if upper_bound - lower_bound < 0.2:
            lower_bound = max(0.0, upper_bound - 0.2)
        ax.set_ylim(lower_bound, upper_bound)
    except Exception:
        # Fallback to full range if anything goes wrong
        ax.set_ylim(0, 1.0)
    
    # Create legend with average scores
    legend_labels = []
    for model_name in model_order:
        avg_score = model_performance[model_name]
        legend_labels.append(f'{model_name} ({avg_score:.3f})')
    
    # Get handles from the plot
    handles, _ = ax.get_legend_handles_labels()
    
    # Legend positioned to avoid covering data
    legend = ax.legend(handles, legend_labels, bbox_to_anchor=(1.02, 1), loc='upper left', 
                      frameon=True, fancybox=False, shadow=False, framealpha=0.9, 
                      edgecolor='#CCCCCC', fontsize=8)
    legend.get_frame().set_linewidth(0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches='tight')
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches='tight')
    plt.close(fig)


def make_latex_table(df: pd.DataFrame, sizes_sorted: List[int]) -> str:
    """Generate LaTeX table for goal achievement metric."""
    # Order models by goal achievement performance (same as chart)
    model_performance = {}
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        avg_goal = model_data["goal_achievement"].mean()
        model_performance[model] = avg_goal
    
    # Sort by performance (worst to best, so best appears on right)
    models = sorted(model_performance.keys(), key=lambda x: model_performance[x])
    
    # Pivot table for goal achievement
    goal_pivot = df.pivot_table(
        index="size", columns="model", values="goal_achievement", aggfunc="mean"
    ).reindex(sizes_sorted)
    
    cols_spec = "|c|" + "|".join(["c" for _ in models]) + "|"
    latex_lines: List[str] = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\caption{Goal Achievement by Sequence Length across Models (higher is better)}")
    latex_lines.append("  \\label{tab:goal_achievement_by_size_across_models}")
    latex_lines.append(f"  \\begin{{tabular}}{{{cols_spec}}}")
    latex_lines.append("    \\hline")
    header = ["\\textbf{Seq. Length}"] + [f"\\textbf{{{m}}}" for m in models]
    latex_lines.append("    " + " & ".join(header) + " \\\\ \\hline")

    for size in sizes_sorted:
        row_vals: List[str] = [str(size)]
        for m in models:
            val = goal_pivot.loc[size, m] if m in goal_pivot.columns else float("nan")
            if pd.isna(val):
                row_vals.append("-")
            else:
                row_vals.append(f"{val:.2f}")
        latex_lines.append("    " + " & ".join(row_vals) + " \\\\ \\hline")

    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table}")
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Goal Achievement by Sequence Length across models",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="List of runs as name:/abs/path. If omitted, defaults are used.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/aggregated_performance").resolve(),
        help="Directory to write outputs (chart and LaTeX)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    runs = (
        parse_runs_arg(args.runs)
        if args.runs
        else default_runs(repo_root)
    )

    # Validate
    for name, path in runs.items():
        if not path.exists():
            print(f"Run for '{name}' not found: {path}")
            continue
        if not path.is_dir():
            print(f"Run path for '{name}' is not a directory: {path}")

    df, sizes_sorted = collect_performance_by_size(runs)

    # Outputs
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate goal achievement chart
    chart_path = out_dir / "goal_achievement_by_size.png"
    make_goal_achievement_chart(df, sizes_sorted, chart_path)

    # Generate LaTeX table
    goal_table = make_latex_table(df, sizes_sorted)
    
    goal_latex_path = out_dir / "goal_achievement_by_size.tex"
    goal_latex_path.write_text(goal_table, encoding="utf-8")

    # Print summary
    print(f"Saved goal achievement chart: {chart_path}")
    print(f"Saved goal achievement LaTeX: {goal_latex_path}")

    # Print quick stats (ordered by goal achievement)
    print("\nGoal Achievement Summary (ordered worst to best):")
    model_performance = {}
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        avg_goal = model_data["goal_achievement"].mean()
        model_performance[model] = avg_goal
    
    # Sort by performance (worst to best)
    sorted_models = sorted(model_performance.keys(), key=lambda x: model_performance[x])
    for model in sorted_models:
        print(f"  {model}: Goal={model_performance[model]:.3f}")


if __name__ == "__main__":
    main()
