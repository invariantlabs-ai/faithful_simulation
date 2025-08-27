#!/usr/bin/env python3
"""
Personality Radar Chart Generator - Creates radar charts showing average trace alignment 
and goal achievement scores by personality types across different models.

Usage: 
    python personality_radar_chart.py results_dir1 [results_dir2 ...]
    python personality_radar_chart.py ./results/gpt-4-1 ./results/gpt-4-1-mini
    python personality_radar_chart.py ./results  # loads all models in results directory
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import colorsys


def load_model_data(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load data from model result folders. Can handle both individual model directories and parent directories containing models."""
    model_data = {}
    
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return model_data
    
    # Check if this directory itself contains big_eval (i.e., it's a model directory)
    big_eval_dir = results_dir / "big_eval"
    if big_eval_dir.exists():
        # This is a model directory itself
        print(f"Loading data for model: {results_dir.name}")
        data = load_single_model_data(big_eval_dir, results_dir.name)
        if data:
            model_data[results_dir.name] = data
    else:
        # Look for model directories within this directory
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # Look for big_eval subdirectory
            big_eval_dir = model_dir / "big_eval"
            if not big_eval_dir.exists():
                continue
                
            print(f"Loading data for model: {model_dir.name}")
            data = load_single_model_data(big_eval_dir, model_dir.name)
            if data:
                model_data[model_dir.name] = data
    
    return model_data


def load_single_model_data(big_eval_dir: Path, model_name: str) -> Dict[str, Any]:
    """Load data from a single model's big_eval directory."""
    # Load conversations, trace alignments, and alignment summary
    conversations = []
    trace_alignments = {}
    alignment_summary = []
    
    # Expected file names (same as conversation viewer)
    conversation_files = ["conversations.json", "conversation_data.json", "conversations_data.json"]
    alignment_files = ["trace_alignments.json", "alignments.json"]
    summary_files = ["alignment_summary.json", "summary.json", "trace_summary.json"]
    
    # Load conversations
    for filename in conversation_files:
        filepath = big_eval_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                print(f"  ✅ Loaded {len(conversations)} conversations from {filename}")
                break
            except Exception as e:
                print(f"  ⚠️ Error loading {filename}: {e}")
    
    # Load trace alignments
    for filename in alignment_files:
        filepath = big_eval_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    trace_alignments = json.load(f)
                print(f"  ✅ Loaded alignments for {len(trace_alignments)} trace sets from {filename}")
                break
            except Exception as e:
                print(f"  ⚠️ Error loading {filename}: {e}")
    
    # Load alignment summary
    for filename in summary_files:
        filepath = big_eval_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    alignment_summary = json.load(f)
                print(f"  ✅ Loaded alignment summary for {len(alignment_summary)} trace sets from {filename}")
                break
            except Exception as e:
                print(f"  ⚠️ Error loading {filename}: {e}")
    
    if conversations or trace_alignments or alignment_summary:
        return {
            'conversations': conversations,
            'trace_alignments': trace_alignments,
            'alignment_summary': alignment_summary
        }
    else:
        print(f"  ❌ No valid data found for {model_name}")
        return {}


def get_similarity_score(conv: Dict[str, Any], trace_alignments: Dict[str, Any] = None) -> float:
    """Get similarity score from alignment data or calculate as fallback."""
    # First try to get from trace alignment data
    if trace_alignments and conv.get("trace_set_id"):
        trace_set_id = conv["trace_set_id"]
        instantiation_id = conv.get("instantiation_id", 0)
        
        if trace_set_id in trace_alignments:
            alignment_data = trace_alignments[trace_set_id]
            alignments = alignment_data.get("alignments", [])
            
            # Find the alignment for this specific instantiation
            if instantiation_id < len(alignments):
                alignment = alignments[instantiation_id]
                return alignment.get("similarity", 0.0)
    
    return 0.0


def get_goal_achievement_score(conv: Dict[str, Any], trace_alignments: Dict[str, Any] = None) -> float:
    """Get goal achievement score for a conversation from trace alignments if available."""
    trace_set_id = conv.get("trace_set_id")
    inst_id = conv.get("instantiation_id", 0)
    if trace_alignments and trace_set_id and trace_set_id in trace_alignments:
        goal_results = trace_alignments[trace_set_id].get("goal_achievement_results", [])
        for result in goal_results:
            if result.get("conversation_id", -1) == conv.get("conversation_id", -1) or result.get("conversation_id", -1) == conv.get("original_index", -1):
                return result.get("score", 0.0)
        # fallback: try by instantiation_id order
        if inst_id < len(goal_results):
            return goal_results[inst_id].get("score", 0.0)
    return 0.0


def calculate_personality_averages(model_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict, Dict, List, List]:
    """Calculate average scores for user and environment personalities across all models."""
    
    # Collect all personalities across all models
    all_user_personalities = set()
    all_env_personalities = set()
    
    for model_name, data in model_data.items():
        conversations = data.get('conversations', [])
        for conv in conversations:
            user_p = conv.get('user_personality_name', 'None') or 'None'
            env_p = conv.get('environment_personality_name', 'None') or 'None'
            all_user_personalities.add(user_p)
            all_env_personalities.add(env_p)
    
    all_user_personalities = sorted(list(all_user_personalities))
    all_env_personalities = sorted(list(all_env_personalities))
    
    print(f"Found personalities:")
    print(f"  User: {all_user_personalities}")
    print(f"  Environment: {all_env_personalities}")
    
    # Calculate averages for each model
    user_averages = {}  # model -> {personality -> {trace_alignment: score, goal_achievement: score}}
    env_averages = {}   # model -> {personality -> {trace_alignment: score, goal_achievement: score}}
    
    for model_name, data in model_data.items():
        conversations = data.get('conversations', [])
        trace_alignments = data.get('trace_alignments', {})
        
        if not conversations:
            continue
        
        print(f"\nCalculating averages for {model_name}...")
        
        # User personality averages (average across all environment personalities for each user personality)
        user_scores = defaultdict(lambda: {'trace_alignment': [], 'goal_achievement': []})
        
        for user_p in all_user_personalities:
            # Get all conversations for this user personality
            user_convs = [c for c in conversations if c.get('user_personality_name', 'None') == user_p]
            
            if user_convs:
                trace_scores = []
                goal_scores = []
                
                for conv in user_convs:
                    trace_score = get_similarity_score(conv, trace_alignments)
                    goal_score = get_goal_achievement_score(conv, trace_alignments)
                    
                    if trace_score > 0 or goal_score > 0:  # Only include if we have valid scores
                        trace_scores.append(trace_score)
                        goal_scores.append(goal_score)
                
                if trace_scores and goal_scores:
                    user_scores[user_p]['trace_alignment'] = np.mean(trace_scores)
                    user_scores[user_p]['goal_achievement'] = np.mean(goal_scores)
                    print(f"  User {user_p}: trace={user_scores[user_p]['trace_alignment']:.3f}, goal={user_scores[user_p]['goal_achievement']:.3f} ({len(trace_scores)} conversations)")
        
        # Environment personality averages (average across all user personalities for each env personality)
        env_scores = defaultdict(lambda: {'trace_alignment': [], 'goal_achievement': []})
        
        for env_p in all_env_personalities:
            # Get all conversations for this environment personality
            env_convs = [c for c in conversations if c.get('environment_personality_name', 'None') == env_p]
            
            if env_convs:
                trace_scores = []
                goal_scores = []
                
                for conv in env_convs:
                    trace_score = get_similarity_score(conv, trace_alignments)
                    goal_score = get_goal_achievement_score(conv, trace_alignments)
                    
                    if trace_score > 0 or goal_score > 0:  # Only include if we have valid scores
                        trace_scores.append(trace_score)
                        goal_scores.append(goal_score)
                
                if trace_scores and goal_scores:
                    env_scores[env_p]['trace_alignment'] = np.mean(trace_scores)
                    env_scores[env_p]['goal_achievement'] = np.mean(goal_scores)
                    print(f"  Env {env_p}: trace={env_scores[env_p]['trace_alignment']:.3f}, goal={env_scores[env_p]['goal_achievement']:.3f} ({len(trace_scores)} conversations)")
        
        user_averages[model_name] = dict(user_scores)
        env_averages[model_name] = dict(env_scores)
    
    return user_averages, env_averages, all_user_personalities, all_env_personalities


def generate_colors(n_models: int) -> List[str]:
    """Generate colors for models using a red-blue color scale."""
    if n_models == 1:
        return ['#8B4513']  # Brown for single model
    
    colors = []
    for i in range(n_models):
        # Generate colors from red to blue
        hue = (240 - i * (240 / (n_models - 1))) / 360  # From red (0) to blue (240)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def create_radar_chart(user_averages: Dict, env_averages: Dict, 
                      user_personalities: List[str], env_personalities: List[str],
                      output_path: str = "personality_radar_chart.png"):
    """Create radar charts showing both user and environment personalities on each chart."""
    
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
    
    # Create figure and axis
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    
    # Get model names and colors
    model_names = list(user_averages.keys())
    colors = generate_colors(len(model_names))
    
    # Left chart: Trace Alignment Scores (both user and environment personalities)
    ax_left.set_title("Trace Alignment Scores", pad=20, fontsize=14, fontweight='bold', color='darkred')
    
    for model_idx, model_name in enumerate(model_names):
        user_data = user_averages[model_name]
        env_data = env_averages[model_name]
        
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
                        label=model_name, color=colors[model_idx], alpha=0.8)
            ax_left.fill(all_angles, all_scores, alpha=0.05, color=colors[model_idx])
    
    # Configure left chart
    ax_left.set_ylim(0, 1)
    ax_left.set_xticks(angles)
    ax_left.set_xticklabels(all_personalities, fontsize=10)
    ax_left.grid(True, color='darkgrey', alpha=1.0, linewidth=1.0)
    ax_left.set_facecolor('white')
    
    # Add vertical divider line and labels
    ax_left.plot([np.pi/2, np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Top vertical line
    ax_left.plot([3*np.pi/2, 3*np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Bottom vertical line
    ax_left.text(np.pi, 0.5, 'User', fontsize=12, fontweight='bold', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax_left.text(0, 0.5, 'Environment', fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax_left.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Right chart: Goal Achievement Scores (both user and environment personalities)
    ax_right.set_title("Goal Achievement Scores", pad=20, fontsize=14, fontweight='bold', color='darkblue')
    
    for model_idx, model_name in enumerate(model_names):
        user_data = user_averages[model_name]
        env_data = env_averages[model_name]
        
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
                         label=model_name, color=colors[model_idx], alpha=0.8)
            ax_right.fill(all_angles, all_scores, alpha=0.05, color=colors[model_idx])
    
    # Configure right chart
    ax_right.set_ylim(0, 1)
    ax_right.set_xticks(angles)
    ax_right.set_xticklabels(all_personalities, fontsize=10)
    ax_right.grid(True, color='darkgrey', alpha=1.0, linewidth=1.0)
    ax_right.set_facecolor('white')
    
    # Add vertical divider line and labels
    ax_right.plot([np.pi/2, np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Top vertical line
    ax_right.plot([3*np.pi/2, 3*np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Bottom vertical line
    ax_right.text(np.pi, 0.5, 'User', fontsize=12, fontweight='bold', ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax_right.text(0, 0.5, 'Environment', fontsize=12, fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax_right.legend(loc='upper left', bbox_to_anchor=(-0.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate personality radar charts from model results")
    parser.add_argument("results_dirs", nargs='+', 
                       help="One or more directories containing model result folders")
    parser.add_argument("-o", "--output", default="personality_radar_chart.png",
                       help="Output filename (default: personality_radar_chart.png)")
    
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
    
    # Calculate personality averages
    user_averages, env_averages, user_personalities, env_personalities = calculate_personality_averages(all_model_data)
    
    if not user_averages and not env_averages:
        print("No personality data found to plot.")
        sys.exit(1)
    
    # Create radar chart
    create_radar_chart(user_averages, env_averages, user_personalities, env_personalities, args.output)


if __name__ == "__main__":
    main()
