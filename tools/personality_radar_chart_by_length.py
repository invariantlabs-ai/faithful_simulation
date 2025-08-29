#!/usr/bin/env python3
"""
Personality Radar Chart by Task Length - Creates radar charts showing average trace alignment 
and goal achievement scores by personality types, broken down by task sequence length.

Usage: 
    python personality_radar_chart_by_length.py results_dir1 [results_dir2 ...]
    python personality_radar_chart_by_length.py ./results/gpt-4-1 ./results/gpt-4-1-mini
    python personality_radar_chart_by_length.py ./results  # loads all models in results directory
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import colorsys

# Import functions from the original script
from personality_radar_chart import (
    load_model_data, 
    load_single_model_data,
    get_similarity_score,
    get_goal_achievement_score
)


def calculate_personality_averages_by_length(model_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict, Dict, List, List, List]:
    """Calculate average scores for user and environment personalities, grouped by task sequence length."""
    
    # Collect all personalities across all models
    all_user_personalities = set()
    all_env_personalities = set()
    all_lengths = set()
    
    for model_name, data in model_data.items():
        conversations = data.get('conversations', [])
        for conv in conversations:
            user_p = conv.get('user_personality_name', 'None') or 'None'
            env_p = conv.get('environment_personality_name', 'None') or 'None'
            all_user_personalities.add(user_p)
            all_env_personalities.add(env_p)
            
            # Calculate sequence length (use expected length from user_source, fallback to used_tools)
            expected_length = len(conv.get('user_source', []))
            actual_length = len(conv.get('used_tools', []))
            seq_length = expected_length if expected_length > 0 else actual_length
            all_lengths.add(seq_length)
    
    all_user_personalities = sorted(list(all_user_personalities))
    all_env_personalities = sorted(list(all_env_personalities))
    all_lengths = sorted(list(all_lengths))
    
    print(f"Found personalities:")
    print(f"  User: {all_user_personalities}")
    print(f"  Environment: {all_env_personalities}")
    print(f"  Task lengths: {all_lengths}")
    
    # Calculate averages for each length, averaged across all models
    user_averages_by_length = {}  # length -> {personality -> {trace_alignment: score, goal_achievement: score}}
    env_averages_by_length = {}   # length -> {personality -> {trace_alignment: score, goal_achievement: score}}
    
    for seq_length in all_lengths:
        print(f"\nCalculating averages for sequence length {seq_length}...")
        
        # Collect all conversations of this length across all models
        length_conversations = []
        length_trace_alignments = {}
        
        for model_name, data in model_data.items():
            conversations = data.get('conversations', [])
            trace_alignments = data.get('trace_alignments', {})
            
            for conv in conversations:
                expected_length = len(conv.get('user_source', []))
                actual_length = len(conv.get('used_tools', []))
                conv_seq_length = expected_length if expected_length > 0 else actual_length
                
                if conv_seq_length == seq_length:
                    length_conversations.append(conv)
                    # Merge trace alignments for this conversation
                    trace_set_id = conv.get('trace_set_id')
                    if trace_set_id and trace_set_id in trace_alignments:
                        length_trace_alignments[trace_set_id] = trace_alignments[trace_set_id]
        
        if not length_conversations:
            continue
            
        # User personality averages for this length
        user_scores = defaultdict(lambda: {'trace_alignment': [], 'goal_achievement': []})
        
        for user_p in all_user_personalities:
            # Get all conversations for this user personality and length
            user_convs = [c for c in length_conversations if c.get('user_personality_name', 'None') == user_p]
            
            if user_convs:
                trace_scores = []
                goal_scores = []
                
                for conv in user_convs:
                    trace_score = get_similarity_score(conv, length_trace_alignments)
                    goal_score = get_goal_achievement_score(conv, length_trace_alignments)
                    
                    if trace_score > 0 or goal_score > 0:  # Only include if we have valid scores
                        trace_scores.append(trace_score)
                        goal_scores.append(goal_score)
                
                if trace_scores and goal_scores:
                    user_scores[user_p]['trace_alignment'] = np.mean(trace_scores)
                    user_scores[user_p]['goal_achievement'] = np.mean(goal_scores)
                    print(f"  User {user_p} (len {seq_length}): trace={user_scores[user_p]['trace_alignment']:.3f}, goal={user_scores[user_p]['goal_achievement']:.3f} ({len(trace_scores)} conversations)")
        
        # Environment personality averages for this length
        env_scores = defaultdict(lambda: {'trace_alignment': [], 'goal_achievement': []})
        
        for env_p in all_env_personalities:
            # Get all conversations for this environment personality and length
            env_convs = [c for c in length_conversations if c.get('environment_personality_name', 'None') == env_p]
            
            if env_convs:
                trace_scores = []
                goal_scores = []
                
                for conv in env_convs:
                    trace_score = get_similarity_score(conv, length_trace_alignments)
                    goal_score = get_goal_achievement_score(conv, length_trace_alignments)
                    
                    if trace_score > 0 or goal_score > 0:  # Only include if we have valid scores
                        trace_scores.append(trace_score)
                        goal_scores.append(goal_score)
                
                if trace_scores and goal_scores:
                    env_scores[env_p]['trace_alignment'] = np.mean(trace_scores)
                    env_scores[env_p]['goal_achievement'] = np.mean(goal_scores)
                    print(f"  Env {env_p} (len {seq_length}): trace={env_scores[env_p]['trace_alignment']:.3f}, goal={env_scores[env_p]['goal_achievement']:.3f} ({len(trace_scores)} conversations)")
        
        user_averages_by_length[seq_length] = dict(user_scores)
        env_averages_by_length[seq_length] = dict(env_scores)
    
    return user_averages_by_length, env_averages_by_length, all_user_personalities, all_env_personalities, all_lengths


def generate_colors_for_lengths(n_lengths: int) -> List[str]:
    """Generate colors for different task lengths using a red-to-blue gradient."""
    if n_lengths == 1:
        return ['#8B4513']  # Brown for single length
    
    colors = []
    for i in range(n_lengths):
        # Generate colors from red to blue (same as original script)
        hue = (240 - i * (240 / (n_lengths - 1))) / 360  # From red (0) to blue (240)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def create_radar_chart_by_length(user_averages_by_length: Dict, env_averages_by_length: Dict, 
                                user_personalities: List[str], env_personalities: List[str],
                                all_lengths: List[int], output_path: str = "personality_radar_chart_by_length.png"):
    """Create radar charts showing both user and environment personalities, broken down by task length."""
    
    # Combine all personalities for the full circle
    all_personalities = user_personalities + env_personalities
    n_personalities = len(all_personalities)
    
    if n_personalities == 0:
        print("No personalities found to plot")
        return
    
    # Calculate angles: left semicircle for users, right semicircle for environments
    # Left semicircle: π/2 to 3π/2 (90° to 270°, top going counterclockwise to bottom)
    if len(user_personalities) > 0:
        user_angles = np.linspace(np.pi/2, 3*np.pi/2, len(user_personalities) + 1, endpoint=False).tolist()[1:]
    else:
        user_angles = []
    
    # Right semicircle: 3π/2 to π/2 (270° to 90°, bottom going counterclockwise to top)
    # This means: 3π/2, 7π/4, 0, π/4, π/2 for the right side
    if len(env_personalities) > 0:
        # Create angles from 3π/2 to 5π/2 then convert to [0, 2π] range
        raw_env_angles = np.linspace(3*np.pi/2, 5*np.pi/2, len(env_personalities) + 1, endpoint=False)[1:]
        env_angles = [(angle % (2*np.pi)) for angle in raw_env_angles]
    else:
        env_angles = []
    
    angles = user_angles + env_angles
    
    # Create figure and axis with larger fonts
    plt.rcParams.update({
        'font.size': 19,
        'axes.titlesize': 26,
        'axes.labelsize': 22,
        'legend.fontsize': 17
    })
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 9.5), subplot_kw=dict(projection='polar'))
    
    # Get length names and colors
    colors = generate_colors_for_lengths(len(all_lengths))
    
    # Left chart: Trace Alignment Scores (both user and environment personalities)
    ax_left.set_title("Trace Alignment Scores\n(by Task Length)", pad=26, fontsize=26, fontweight='bold', color='darkred')
    
    for length_idx, seq_length in enumerate(all_lengths):
        user_data = user_averages_by_length.get(seq_length, {})
        env_data = env_averages_by_length.get(seq_length, {})
        
        # Prepare data for all personalities
        all_scores = []
        all_angles = []
        
        # Add user personality scores (left semicircle)
        for i, personality in enumerate(user_personalities):
            if personality in user_data:
                all_scores.append(user_data[personality].get('trace_alignment', 0))
                all_angles.append(user_angles[i])
        
        # Add environment personality scores (right semicircle)
        for i, personality in enumerate(env_personalities):
            if personality in env_data:
                all_scores.append(env_data[personality].get('trace_alignment', 0))
                all_angles.append(env_angles[i])
        
        if all_scores:
            # Close the polygon
            all_scores += [all_scores[0]]
            all_angles += [all_angles[0]]
            
            # Plot the data
            ax_left.plot(all_angles, all_scores, 'o-', linewidth=2, 
                        label=f"Length {seq_length}", color=colors[length_idx], alpha=0.8)
            ax_left.fill(all_angles, all_scores, alpha=0.15, color=colors[length_idx])
    
    # Configure left chart
    ax_left.set_ylim(0, 1)
    ax_left.set_xticks(angles)
    ax_left.set_xticklabels(all_personalities, fontsize=17)
    ax_left.grid(True, color='darkgrey', alpha=1.0, linewidth=1.0)
    ax_left.set_facecolor('white')
    
    # Add vertical divider line and labels
    ax_left.plot([np.pi/2, np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Top vertical line
    ax_left.plot([3*np.pi/2, 3*np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Bottom vertical line
    ax_left.text(np.pi, 0.5, 'User', fontsize=19, fontweight='bold', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax_left.text(0, 0.5, 'Environment', fontsize=19, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    # Wrap tick labels with boxes for readability
    for lbl in ax_left.get_xticklabels():
        lbl.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.85))
    
    ax_left.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    
    # Right chart: Goal Achievement Scores (both user and environment personalities)
    ax_right.set_title("Goal Achievement Scores\n(by Task Length)", pad=26, fontsize=26, fontweight='bold', color='darkblue')
    
    for length_idx, seq_length in enumerate(all_lengths):
        user_data = user_averages_by_length.get(seq_length, {})
        env_data = env_averages_by_length.get(seq_length, {})
        
        # Prepare data for all personalities
        all_scores = []
        all_angles = []
        
        # Add user personality scores (left semicircle)
        for i, personality in enumerate(user_personalities):
            if personality in user_data:
                all_scores.append(user_data[personality].get('goal_achievement', 0))
                all_angles.append(user_angles[i])
        
        # Add environment personality scores (right semicircle)
        for i, personality in enumerate(env_personalities):
            if personality in env_data:
                all_scores.append(env_data[personality].get('goal_achievement', 0))
                all_angles.append(env_angles[i])
        
        if all_scores:
            # Close the polygon
            all_scores += [all_scores[0]]
            all_angles += [all_angles[0]]
            
            # Plot the data
            ax_right.plot(all_angles, all_scores, 'o-', linewidth=2, 
                         label=f"Length {seq_length}", color=colors[length_idx], alpha=0.8)
            ax_right.fill(all_angles, all_scores, alpha=0.15, color=colors[length_idx])
    
    # Configure right chart
    ax_right.set_ylim(0, 1)
    ax_right.set_xticks(angles)
    ax_right.set_xticklabels(all_personalities, fontsize=17)
    ax_right.grid(True, color='darkgrey', alpha=1.0, linewidth=1.0)
    ax_right.set_facecolor('white')
    
    # Add vertical divider line and labels
    ax_right.plot([np.pi/2, np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Top vertical line
    ax_right.plot([3*np.pi/2, 3*np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Bottom vertical line
    ax_right.text(np.pi, 0.5, 'User', fontsize=19, fontweight='bold', ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax_right.text(0, 0.5, 'Environment', fontsize=19, fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    # Wrap tick labels with boxes for readability
    for lbl in ax_right.get_xticklabels():
        lbl.set_bbox(dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.85))
    
    # Legend only on the left plot to avoid duplication
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.04)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart by task length saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate personality radar charts by task length from model results")
    parser.add_argument("results_dirs", nargs='+', 
                       help="One or more directories containing model result folders")
    parser.add_argument("-o", "--output", default="personality_radar_chart_by_length.png",
                       help="Output filename (default: personality_radar_chart_by_length.png)")
    
    args = parser.parse_args()
    
    # Load data from all result directories
    all_model_data = {}
    
    for results_dir_str in args.results_dirs:
        results_dir = Path(results_dir_str)
        print(f"Loading data from: {results_dir}")
        
        model_data = load_model_data(results_dir)
        
        # Merge model data, handling potential name conflicts
        for model_name, data in model_data.items():
            # If model name already exists, append directory name
            final_model_name = model_name
            if final_model_name in all_model_data:
                final_model_name = f"{model_name}_{results_dir.name}"
            
            all_model_data[final_model_name] = data
    
    if not all_model_data:
        print("No model data found. Make sure the results directories contain model folders with big_eval subdirectories.")
        sys.exit(1)
    
    print(f"\nLoaded data for {len(all_model_data)} models: {list(all_model_data.keys())}")
    
    # Calculate personality averages by task length
    user_averages_by_length, env_averages_by_length, user_personalities, env_personalities, all_lengths = calculate_personality_averages_by_length(all_model_data)
    
    if not user_averages_by_length and not env_averages_by_length:
        print("No personality data found to plot.")
        sys.exit(1)
    
    # Create radar chart
    create_radar_chart_by_length(user_averages_by_length, env_averages_by_length, 
                                user_personalities, env_personalities, all_lengths, args.output)


if __name__ == "__main__":
    main()
