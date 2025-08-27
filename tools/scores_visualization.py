#!/usr/bin/env python3
"""
Scores Visualization Script

Creates circular/sun-like bar charts for model benchmark results showing:
1. Goal Achievement rates
2. Trace Alignment scores

The data is hardcoded with averaged scores across user personalities and environment archetypes.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# Hardcoded benchmark data - averaged scores across all user personalities and environment archetypes
GOAL_ACHIEVEMENT_DATA = {
    'gpt-4o-mini': 0.786,
    'gpt-4': 0.825,  # Example data - replace with actual values
    'claude-3.5-sonnet': 0.798,  # Example data - replace with actual values
    'gemini-1.5-pro': 0.742,  # Example data - replace with actual values
}

TRACE_ALIGNMENT_DATA = {
    'gpt-4o-mini': 0.572,
    'gpt-4': 0.634,  # Example data - replace with actual values
    'claude-3.5-sonnet': 0.591,  # Example data - replace with actual values
    'gemini-1.5-pro': 0.523,  # Example data - replace with actual values
}

def create_circular_bar_chart(data: Dict[str, float], title: str, metric_name: str, 
                            save_path: str = None, figsize: Tuple[int, int] = (10, 10)):
    """
    Create a circular/sun-like bar chart for the given data.
    
    Args:
        data: Dictionary mapping model names to their scores
        title: Title for the chart
        metric_name: Name of the metric being visualized
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    # Set up the figure and polar axis
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Prepare data
    models = list(data.keys())
    scores = list(data.values())
    n_models = len(models)
    
    # Calculate angles for each bar (evenly distributed around the circle)
    theta = np.linspace(0, 2 * np.pi, n_models, endpoint=False)
    
    # Width of each bar
    width = 2 * np.pi / n_models * 0.8  # 0.8 to leave some space between bars
    
    # Create color map - use a colorful palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    # Create the bars
    bars = ax.bar(theta, scores, width=width, bottom=0.0, alpha=0.8)
    
    # Color the bars
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
        bar.set_edgecolor('black')
        bar.set_linewidth(1)
    
    # Customize the chart
    ax.set_theta_zero_location('N')  # Start from top
    ax.set_theta_direction(-1)  # Clockwise
    
    # Set radial limits based on the metric
    if 'goal' in title.lower() or 'achievement' in title.lower():
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    else:  # trace alignment
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Add model names as labels
    ax.set_xticks(theta)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    
    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add score labels on top of bars
    for angle, score, model in zip(theta, scores, models):
        ax.text(angle, score + 0.05, f'{score:.3f}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                rotation=np.degrees(angle) - 90 if np.degrees(angle) > 90 else np.degrees(angle) + 90)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    return fig, ax

def create_comparison_chart(goal_data: Dict[str, float], alignment_data: Dict[str, float], 
                          save_path: str = None, figsize: Tuple[int, int] = (16, 8)):
    """
    Create a side-by-side comparison of both metrics.
    
    Args:
        goal_data: Goal achievement data
        alignment_data: Trace alignment data
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Create both charts
    models = list(goal_data.keys())
    n_models = len(models)
    theta = np.linspace(0, 2 * np.pi, n_models, endpoint=False)
    width = 2 * np.pi / n_models * 0.8
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    # Goal Achievement chart
    goal_scores = list(goal_data.values())
    bars1 = ax1.bar(theta, goal_scores, width=width, bottom=0.0, alpha=0.8)
    for bar, color in zip(bars1, colors):
        bar.set_facecolor(color)
        bar.set_edgecolor('black')
        bar.set_linewidth(1)
    
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax1.set_xticks(theta)
    ax1.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax1.set_title('Goal Achievement', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Add score labels
    for angle, score in zip(theta, goal_scores):
        ax1.text(angle, score + 0.05, f'{score:.3f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Trace Alignment chart
    alignment_scores = list(alignment_data.values())
    bars2 = ax2.bar(theta, alignment_scores, width=width, bottom=0.0, alpha=0.8)
    for bar, color in zip(bars2, colors):
        bar.set_facecolor(color)
        bar.set_edgecolor('black')
        bar.set_linewidth(1)
    
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax2.set_xticks(theta)
    ax2.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax2.set_title('Trace Alignment', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    # Add score labels
    for angle, score in zip(theta, alignment_scores):
        ax2.text(angle, score + 0.05, f'{score:.3f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")
    
    return fig, (ax1, ax2)

def main():
    """Main function to generate all visualizations."""
    print("Generating scores visualization charts...")
    
    # Create individual charts
    print("\n1. Creating Goal Achievement chart...")
    fig1, ax1 = create_circular_bar_chart(
        GOAL_ACHIEVEMENT_DATA, 
        'Goal Achievement by Model', 
        'Goal Achievement Rate',
        'goal_achievement_circular.png'
    )
    
    print("2. Creating Trace Alignment chart...")
    fig2, ax2 = create_circular_bar_chart(
        TRACE_ALIGNMENT_DATA, 
        'Trace Alignment by Model', 
        'Trace Alignment Score',
        'trace_alignment_circular.png'
    )
    
    print("3. Creating comparison chart...")
    fig3, (ax3, ax4) = create_comparison_chart(
        GOAL_ACHIEVEMENT_DATA,
        TRACE_ALIGNMENT_DATA,
        'model_comparison_circular.png'
    )
    
    # Display summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print("\nGoal Achievement:")
    for model, score in GOAL_ACHIEVEMENT_DATA.items():
        print(f"  {model:<20}: {score:.3f}")
    
    print(f"\n  Average: {np.mean(list(GOAL_ACHIEVEMENT_DATA.values())):.3f}")
    print(f"  Best: {max(GOAL_ACHIEVEMENT_DATA.keys(), key=GOAL_ACHIEVEMENT_DATA.get)} ({max(GOAL_ACHIEVEMENT_DATA.values()):.3f})")
    
    print("\nTrace Alignment:")
    for model, score in TRACE_ALIGNMENT_DATA.items():
        print(f"  {model:<20}: {score:.3f}")
    
    print(f"\n  Average: {np.mean(list(TRACE_ALIGNMENT_DATA.values())):.3f}")
    print(f"  Best: {max(TRACE_ALIGNMENT_DATA.keys(), key=TRACE_ALIGNMENT_DATA.get)} ({max(TRACE_ALIGNMENT_DATA.values()):.3f})")
    
    # Show all plots
    plt.show()
    
    print("\nVisualization complete! Charts saved as PNG files.")

if __name__ == "__main__":
    main()
