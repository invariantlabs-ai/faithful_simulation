#!/usr/bin/env python3
"""
Trace-Goal Correlation Analysis - Analyzes the correlation between trace alignment 
and goal achievement to answer: Does faithfulness to ground-truth traces act as a 
causal mediator for end-to-end success, or can agents reach goals via alternative 
but still safe sequences?

Usage: 
    python trace_goal_correlation_analysis.py results_dir1 [results_dir2 ...]
    python trace_goal_correlation_analysis.py ./results/gpt-4-1 ./results/gpt-4-1-mini
    python trace_goal_correlation_analysis.py ./results  # loads all models in results directory
"""

import json
import numpy as np
from pathlib import Path
import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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


def extract_correlation_data(model_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Extract trace alignment and goal achievement scores for all conversations across all models."""
    
    data_rows = []
    
    for model_name, data in model_data.items():
        conversations = data.get('conversations', [])
        trace_alignments = data.get('trace_alignments', {})
        
        if not conversations:
            continue
        
        print(f"Extracting data for {model_name}...")
        
        for conv in conversations:
            trace_score = get_similarity_score(conv, trace_alignments)
            goal_score = get_goal_achievement_score(conv, trace_alignments)
            
            # Only include conversations with valid scores
            if trace_score >= 0 and goal_score >= 0:
                user_personality = conv.get('user_personality_name', 'None') or 'None'
                env_personality = conv.get('environment_personality_name', 'None') or 'None'
                
                data_rows.append({
                    'model': model_name,
                    'conversation_id': conv.get('conversation_id', conv.get('original_index', -1)),
                    'trace_set_id': conv.get('trace_set_id', ''),
                    'user_personality': user_personality,
                    'environment_personality': env_personality,
                    'trace_alignment': trace_score,
                    'goal_achievement': goal_score
                })
    
    df = pd.DataFrame(data_rows)
    print(f"Extracted {len(df)} valid data points across {df['model'].nunique()} models")
    return df


def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate various correlation statistics."""
    
    correlations = {}
    
    # Overall correlation across all models and personalities
    if len(df) > 1:
        pearson_r, pearson_p = pearsonr(df['trace_alignment'], df['goal_achievement'])
        spearman_r, spearman_p = spearmanr(df['trace_alignment'], df['goal_achievement'])
        
        correlations['overall'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_points': len(df)
        }
    
    # Correlations by model
    correlations['by_model'] = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        if len(model_df) > 1:
            try:
                pearson_r, pearson_p = pearsonr(model_df['trace_alignment'], model_df['goal_achievement'])
                spearman_r, spearman_p = spearmanr(model_df['trace_alignment'], model_df['goal_achievement'])
                
                correlations['by_model'][model] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_points': len(model_df)
                }
            except:
                correlations['by_model'][model] = {'error': 'Could not calculate correlation'}
    
    # Correlations by user personality
    correlations['by_user_personality'] = {}
    for personality in df['user_personality'].unique():
        pers_df = df[df['user_personality'] == personality]
        if len(pers_df) > 1:
            try:
                pearson_r, pearson_p = pearsonr(pers_df['trace_alignment'], pers_df['goal_achievement'])
                spearman_r, spearman_p = spearmanr(pers_df['trace_alignment'], pers_df['goal_achievement'])
                
                correlations['by_user_personality'][personality] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_points': len(pers_df)
                }
            except:
                correlations['by_user_personality'][personality] = {'error': 'Could not calculate correlation'}
    
    # Correlations by environment personality
    correlations['by_env_personality'] = {}
    for personality in df['environment_personality'].unique():
        pers_df = df[df['environment_personality'] == personality]
        if len(pers_df) > 1:
            try:
                pearson_r, pearson_p = pearsonr(pers_df['trace_alignment'], pers_df['goal_achievement'])
                spearman_r, spearman_p = spearmanr(pers_df['trace_alignment'], pers_df['goal_achievement'])
                
                correlations['by_env_personality'][personality] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_points': len(pers_df)
                }
            except:
                correlations['by_env_personality'][personality] = {'error': 'Could not calculate correlation'}
    
    return correlations


def generate_model_archetype_matrices(df: pd.DataFrame) -> List[str]:
    """Generate separate matrix tables for each model showing mean trace alignment and goal achievement values."""
    
    latex_output = []
    
    # Get unique values for each category
    models = df['model'].unique()
    user_archetypes = sorted(df['user_personality'].unique())
    env_archetypes = sorted(df['environment_personality'].unique())
    
    # Filter to main models (exclude duplicates/temp versions)
    main_models = []
    for model in models:
        if not any(x in model.lower() for x in ['tmp_', '_claude4sonnet_assessed', '_assessed']):
            main_models.append(model)
    
    # Take first 3 models if we have more than 3
    main_models = sorted(main_models)[:3]
    
    # Generate two matrix tables for each main model (trace alignment and goal achievement)
    for i, model in enumerate(main_models):
        model_clean = model.replace('_', '\\_')  # Escape underscores for LaTeX
        
        # Create column specification (l for first column + c for each environment archetype)
        col_spec = 'l' + 'c' * len(env_archetypes)
        
        # Table 1: Trace Alignment Values
        latex_output.append("\\begin{table}[htbp]")
        latex_output.append("\\centering")
        latex_output.append(f"\\caption{{Trace Alignment Values: {model_clean}}}")
        latex_output.append(f"\\label{{tab:trace_alignment_{model.replace('-', '_').replace('.', '_')}}}")
        latex_output.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_output.append("\\toprule")
        
        # Header row
        header = "User Archetype"
        for env_archetype in env_archetypes:
            env_clean = env_archetype.replace('_', '\\_')
            header += f" & {env_clean}"
        header += " \\\\"
        latex_output.append(header)
        latex_output.append("\\midrule")
        
        # Data rows for trace alignment
        for user_archetype in user_archetypes:
            user_clean = user_archetype.replace('_', '\\_')
            row = user_clean
            
            for env_archetype in env_archetypes:
                # Get mean trace alignment for this specific combination
                subset_df = df[(df['model'] == model) & 
                             (df['user_personality'] == user_archetype) & 
                             (df['environment_personality'] == env_archetype)]
                
                if len(subset_df) > 0:
                    mean_trace = subset_df['trace_alignment'].mean()
                    row += f" & {mean_trace:.3f}"
                else:
                    row += " & --"
            
            row += " \\\\"
            latex_output.append(row)
        
        latex_output.append("\\bottomrule")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")
        
        latex_output.append("")
        
        # Table 2: Goal Achievement Values
        latex_output.append("\\begin{table}[htbp]")
        latex_output.append("\\centering")
        latex_output.append(f"\\caption{{Goal Achievement Values: {model_clean}}}")
        latex_output.append(f"\\label{{tab:goal_achievement_{model.replace('-', '_').replace('.', '_')}}}")
        latex_output.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_output.append("\\toprule")
        
        # Header row
        header = "User Archetype"
        for env_archetype in env_archetypes:
            env_clean = env_archetype.replace('_', '\\_')
            header += f" & {env_clean}"
        header += " \\\\"
        latex_output.append(header)
        latex_output.append("\\midrule")
        
        # Data rows for goal achievement
        for user_archetype in user_archetypes:
            user_clean = user_archetype.replace('_', '\\_')
            row = user_clean
            
            for env_archetype in env_archetypes:
                # Get mean goal achievement for this specific combination
                subset_df = df[(df['model'] == model) & 
                             (df['user_personality'] == user_archetype) & 
                             (df['environment_personality'] == env_archetype)]
                
                if len(subset_df) > 0:
                    mean_goal = subset_df['goal_achievement'].mean()
                    row += f" & {mean_goal:.3f}"
                else:
                    row += " & --"
            
            row += " \\\\"
            latex_output.append(row)
        
        latex_output.append("\\bottomrule")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")
        
        # Add spacing between model sets (except after the last one)
        if i < len(main_models) - 1:
            latex_output.append("")
            latex_output.append("")
    
    return latex_output


def generate_latex_tables(df: pd.DataFrame, correlations: Dict[str, Any], output_prefix: str = "trace_goal_correlation"):
    """Generate LaTeX tables showing Pearson correlation analysis results."""
    
    latex_output = []
    
    # First table - comprehensive correlation analysis
    latex_output.append("\\begin{table}[htbp]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Trace–Goal Correlation Analysis}")
    latex_output.append("\\label{tab:trace_goal_correlation}")
    latex_output.append("\\begin{tabular}{lc}")
    latex_output.append("\\toprule")
    latex_output.append("Category & Pearson $r$ \\\\")
    latex_output.append("\\midrule")
    
    # Overall correlation
    if 'overall' in correlations:
        overall = correlations['overall']
        r = overall['pearson_r']
        latex_output.append(f"Overall & {r:.3f} \\\\")
        latex_output.append("\\midrule")
    
    # Models
    latex_output.append("\\multicolumn{2}{l}{\\textbf{By Model}} \\\\")
    for model, corr_data in correlations['by_model'].items():
        if 'pearson_r' in corr_data:
            r = corr_data['pearson_r']
            model_clean = model.replace('_', '\\_')  # Escape underscores for LaTeX
            latex_output.append(f"\\quad {model_clean} & {r:.3f} \\\\")
        else:
            model_clean = model.replace('_', '\\_')
            latex_output.append(f"\\quad {model_clean} & -- \\\\")
    
    latex_output.append("\\midrule")
    
    # User personalities
    latex_output.append("\\multicolumn{2}{l}{\\textbf{By User Personality}} \\\\")
    for personality, corr_data in correlations['by_user_personality'].items():
        if 'pearson_r' in corr_data:
            r = corr_data['pearson_r']
            personality_clean = personality.replace('_', '\\_')  # Escape underscores for LaTeX
            latex_output.append(f"\\quad {personality_clean} & {r:.3f} \\\\")
        else:
            personality_clean = personality.replace('_', '\\_')
            latex_output.append(f"\\quad {personality_clean} & -- \\\\")
    
    latex_output.append("\\midrule")
    
    # Environment personalities
    latex_output.append("\\multicolumn{2}{l}{\\textbf{By Environment Personality}} \\\\")
    for personality, corr_data in correlations['by_env_personality'].items():
        if 'pearson_r' in corr_data:
            r = corr_data['pearson_r']
            personality_clean = personality.replace('_', '\\_')  # Escape underscores for LaTeX
            latex_output.append(f"\\quad {personality_clean} & {r:.3f} \\\\")
        else:
            personality_clean = personality.replace('_', '\\_')
            latex_output.append(f"\\quad {personality_clean} & -- \\\\")
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    
    # Add space between tables
    latex_output.append("")
    latex_output.append("")
    
    # Matrix tables - one for each main model
    latex_output.extend(generate_model_archetype_matrices(df))
    
    # Save LaTeX output to file
    latex_filename = f"{output_prefix}_tables.tex"
    with open(latex_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_output))
    
    print(f"LaTeX tables saved to: {latex_filename}")
    
    return latex_output


def print_correlation_summary(correlations: Dict[str, Any]):
    """Print a simplified summary of correlation analysis results."""
    
    print("\n" + "="*60)
    print("TRACE-GOAL CORRELATION ANALYSIS")
    print("="*60)
    
    # Overall correlation
    if 'overall' in correlations:
        overall = correlations['overall']
        r = overall['pearson_r']
        p = overall['pearson_p']
        n = overall['n_points']
        
        # Significance indicator
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""
        
        print(f"\nOverall correlation: r = {r:.3f}{sig} (n = {n})")
        
        # Brief interpretation
        if abs(r) > 0.5 and p < 0.05:
            print("→ Strong mediation: trace alignment predicts goal achievement")
        elif abs(r) > 0.3 and p < 0.05:
            print("→ Moderate mediation: some alternative pathways exist")
        elif abs(r) > 0.1 and p < 0.05:
            print("→ Weak mediation: many alternative pathways exist")
        else:
            print("→ No mediation: trace alignment doesn't predict success")
    
    print(f"\n{'Category':<25} {'r':<8} {'n':<5}")
    print("-" * 40)
    
    # Models
    print("BY MODEL:")
    for model, corr_data in correlations['by_model'].items():
        if 'pearson_r' in corr_data:
            r = corr_data['pearson_r']
            p = corr_data['pearson_p']
            n = corr_data['n_points']
            
            # Significance indicator
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = ""
            
            print(f"  {model:<23} {r:.3f}{sig:<3} {n:<5}")
        else:
            print(f"  {model:<23} {'--':<6} {'--':<5}")
    
    # User personalities
    print("\nBY USER PERSONALITY:")
    for personality, corr_data in correlations['by_user_personality'].items():
        if 'pearson_r' in corr_data:
            r = corr_data['pearson_r']
            p = corr_data['pearson_p']
            n = corr_data['n_points']
            
            # Significance indicator
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = ""
            
            print(f"  {personality:<23} {r:.3f}{sig:<3} {n:<5}")
        else:
            print(f"  {personality:<23} {'--':<6} {'--':<5}")
    
    # Environment personalities
    print("\nBY ENVIRONMENT PERSONALITY:")
    for personality, corr_data in correlations['by_env_personality'].items():
        if 'pearson_r' in corr_data:
            r = corr_data['pearson_r']
            p = corr_data['pearson_p']
            n = corr_data['n_points']
            
            # Significance indicator
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = ""
            
            print(f"  {personality:<23} {r:.3f}{sig:<3} {n:<5}")
        else:
            print(f"  {personality:<23} {'--':<6} {'--':<5}")
    
    print(f"\nSignificance: * p < 0.05, ** p < 0.01, *** p < 0.001")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between trace alignment and goal achievement")
    parser.add_argument("results_dirs", nargs='+', 
                       help="One or more directories containing model result folders")
    parser.add_argument("-o", "--output", default="trace_goal_correlation",
                       help="Output filename prefix (default: trace_goal_correlation)")
    parser.add_argument("--save-data", action="store_true",
                       help="Save extracted data to CSV file")
    
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
    
    # Extract correlation data
    df = extract_correlation_data(all_model_data)
    
    if len(df) == 0:
        print("No valid data points found for correlation analysis.")
        sys.exit(1)
    
    # Save data if requested
    if args.save_data:
        csv_filename = f"{args.output}_data.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to: {csv_filename}")
    
    # Calculate correlations
    correlations = calculate_correlations(df)
    
    # Generate LaTeX tables
    generate_latex_tables(df, correlations, args.output)
    
    # Print summary
    print_correlation_summary(correlations)


if __name__ == "__main__":
    main()
