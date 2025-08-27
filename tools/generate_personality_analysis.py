#!/usr/bin/env python3
"""
Personality Analysis Generator - Generate user vs environment comparison tables
and sequence length breakdowns from conversation data.

Usage: python generate_personality_analysis.py <results_directory>

Input: Directory containing conversation data files (conversations.json, trace_alignments.json, etc.)
Output: Formatted tables printed to console
"""

import json
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import statistics


def load_files_from_folder(folder_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Load conversation files automatically from a folder by expected names."""
    conversations = []
    trace_alignments = {}
    alignment_summary = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return conversations, trace_alignments, alignment_summary
    
    # Expected file names
    conversation_files = ["conversations.json", "conversation_data.json", "conversations_data.json"]
    alignment_files = ["trace_alignments.json", "alignments.json"]
    summary_files = ["alignment_summary.json", "summary.json", "trace_summary.json"]
    
    # Load conversations
    for filename in conversation_files:
        filepath = folder / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                print(f"✅ Loaded {len(conversations)} conversations from {filename}")
                break
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
    else:
        print(f"📁 No conversation file found in {folder_path}. Looking for: {', '.join(conversation_files)}")
    
    # Load trace alignments
    for filename in alignment_files:
        filepath = folder / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    trace_alignments = json.load(f)
                print(f"✅ Loaded alignments for {len(trace_alignments)} trace sets from {filename}")
                break
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
    
    # Load alignment summary
    for filename in summary_files:
        filepath = folder / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    alignment_summary = json.load(f)
                print(f"✅ Loaded alignment summary for {len(alignment_summary)} trace sets from {filename}")
                break
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
    
    return conversations, trace_alignments, alignment_summary


def similarity_metric(seq1: List[str], seq2: List[str]) -> float:
    """Calculate normalized Levenshtein Distance similarity between two sequences."""
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    
    # Calculate Levenshtein distance using dynamic programming
    len1, len2 = len(seq1), len(seq2)
    
    # Create a matrix to store distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deletion
                    dp[i][j-1],    # Insertion
                    dp[i-1][j-1]   # Substitution
                )
    
    # Get the Levenshtein distance
    distance = dp[len1][len2]
    
    # Normalize to similarity score (0-1 range)
    max_len = max(len1, len2)
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    
    return similarity


def get_similarity_score(conv: Dict[str, Any], trace_alignments: Dict[str, Any] = None) -> Tuple[float, str]:
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
                return alignment.get("similarity", 0.0), "Weighted Levenshtein"
    
    # Fallback to basic similarity calculation
    user_source_tools = [tool.get('name', '') for tool in conv.get('user_source', [])]
    used_tools = conv.get('used_tools', [])
    similarity = similarity_metric(user_source_tools, used_tools)
    return similarity, "Basic Levenshtein"


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


def generate_user_vs_env_analysis(conversations: List[Dict[str, Any]], trace_alignments: Dict[str, Any]) -> None:
    """Generate user vs environment personality comparison tables."""
    print("🔄 Generating user vs environment analysis...")
    
    # Get unique personalities
    user_personalities = list(set(conv.get('user_personality_name', 'None') for conv in conversations if conv.get('user_personality_name')))
    env_personalities = list(set(conv.get('environment_personality_name', 'None') for conv in conversations if conv.get('environment_personality_name')))
    
    print(f"Found {len(user_personalities)} user personalities and {len(env_personalities)} environment personalities")
    
    # Generate metrics for all combinations
    metrics_data = []
    
    for user_p in user_personalities:
        for env_p in env_personalities:
            # Filter conversations for this personality combination
            filtered_convs = [c for c in conversations 
                            if c.get('user_personality_name', 'None') == user_p 
                            and c.get('environment_personality_name', 'None') == env_p]
            
            if not filtered_convs:
                continue
            
            # Calculate metrics
            goal_scores = [get_goal_achievement_score(c, trace_alignments) for c in filtered_convs]
            alignment_scores = [get_similarity_score(c, trace_alignments)[0] for c in filtered_convs]
            
            avg_goal = sum(goal_scores) / len(goal_scores) if goal_scores else 0
            avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
            std_goal = statistics.stdev(goal_scores) if len(goal_scores) > 1 else 0
            std_alignment = statistics.stdev(alignment_scores) if len(alignment_scores) > 1 else 0
            
            metrics_data.append({
                'User_Personality': user_p,
                'Environment_Personality': env_p,
                'Conversation_Count': len(filtered_convs),
                'Mean_Goal_Achievement': round(avg_goal, 4),
                'Std_Goal_Achievement': round(std_goal, 4),
                'Mean_Alignment_Score': round(avg_alignment, 4),
                'Std_Alignment_Score': round(std_alignment, 4),
                'Min_Goal_Achievement': round(min(goal_scores) if goal_scores else 0, 4),
                'Max_Goal_Achievement': round(max(goal_scores) if goal_scores else 0, 4),
                'Min_Alignment_Score': round(min(alignment_scores) if alignment_scores else 0, 4),
                'Max_Alignment_Score': round(max(alignment_scores) if alignment_scores else 0, 4)
            })
    
    if metrics_data:
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Print detailed metrics
        print("\n" + "="*80)
        print("📊 USER VS ENVIRONMENT DETAILED METRICS")
        print("="*80)
        print(df.to_string(index=False))
        
        # Create pivot tables
        goal_pivot = df.pivot_table(
            index='User_Personality', 
            columns='Environment_Personality', 
            values='Mean_Goal_Achievement',
            aggfunc='mean'
        )
        
        alignment_pivot = df.pivot_table(
            index='User_Personality',
            columns='Environment_Personality',
            values='Mean_Alignment_Score',
            aggfunc='mean'
        )
        
        # Print pivot tables
        print("\n" + "="*60)
        print("🎯 GOAL ACHIEVEMENT PIVOT TABLE")
        print("="*60)
        print(goal_pivot.round(4).to_string())
        
        print("\n" + "="*60)
        print("📏 ALIGNMENT SCORE PIVOT TABLE") 
        print("="*60)
        print(alignment_pivot.round(4).to_string())
        
        # Generate and print summary statistics
        best_goal_idx = df['Mean_Goal_Achievement'].idxmax()
        worst_goal_idx = df['Mean_Goal_Achievement'].idxmin()
        best_alignment_idx = df['Mean_Alignment_Score'].idxmax()
        worst_alignment_idx = df['Mean_Alignment_Score'].idxmin()
        
        print("\n" + "="*60)
        print("📈 SUMMARY STATISTICS")
        print("="*60)
        print(f"Total Personality Combinations: {len(df)}")
        print(f"Total Conversations: {df['Conversation_Count'].sum()}")
        print(f"Overall Mean Goal Achievement: {df['Mean_Goal_Achievement'].mean():.4f}")
        print(f"Overall Mean Alignment Score: {df['Mean_Alignment_Score'].mean():.4f}")
        print()
        print("🏆 BEST COMBINATIONS:")
        print(f"  Goal Achievement: {df.loc[best_goal_idx, 'User_Personality']} + {df.loc[best_goal_idx, 'Environment_Personality']} ({df.loc[best_goal_idx, 'Mean_Goal_Achievement']:.4f})")
        print(f"  Alignment Score:  {df.loc[best_alignment_idx, 'User_Personality']} + {df.loc[best_alignment_idx, 'Environment_Personality']} ({df.loc[best_alignment_idx, 'Mean_Alignment_Score']:.4f})")
        print()
        print("⚠️  WORST COMBINATIONS:")
        print(f"  Goal Achievement: {df.loc[worst_goal_idx, 'User_Personality']} + {df.loc[worst_goal_idx, 'Environment_Personality']} ({df.loc[worst_goal_idx, 'Mean_Goal_Achievement']:.4f})")
        print(f"  Alignment Score:  {df.loc[worst_alignment_idx, 'User_Personality']} + {df.loc[worst_alignment_idx, 'Environment_Personality']} ({df.loc[worst_alignment_idx, 'Mean_Alignment_Score']:.4f})")
    
    else:
        print("⚠️ No data found for user vs environment analysis")


def generate_sequence_length_analysis(conversations: List[Dict[str, Any]], trace_alignments: Dict[str, Any]) -> None:
    """Generate sequence length breakdown analysis."""
    print("🔄 Generating sequence length analysis...")
    
    # Get unique personalities
    user_personalities = list(set(conv.get('user_personality_name', 'None') for conv in conversations if conv.get('user_personality_name')))
    env_personalities = list(set(conv.get('environment_personality_name', 'None') for conv in conversations if conv.get('environment_personality_name')))
    
    # Calculate sequence length data for user vs environment combinations
    sequence_length_data = []
    
    for user_p in user_personalities:
        for env_p in env_personalities:
            # Filter conversations for this personality combination
            filtered_convs = [c for c in conversations 
                            if c.get('user_personality_name', 'None') == user_p 
                            and c.get('environment_personality_name', 'None') == env_p]
            
            if not filtered_convs:
                continue
            
            # Group by sequence length
            length_groups = defaultdict(list)
            for conv in filtered_convs:
                # Use the expected sequence length (from user_source) or actual sequence length (from used_tools)
                expected_length = len(conv.get('user_source', []))
                actual_length = len(conv.get('used_tools', []))
                # Use expected length if available, otherwise actual length
                seq_length = expected_length if expected_length > 0 else actual_length
                length_groups[seq_length].append(conv)
            
            # Calculate metrics for each length group
            for seq_length, length_convs in length_groups.items():
                goal_scores = [get_goal_achievement_score(c, trace_alignments) for c in length_convs]
                alignment_scores = [get_similarity_score(c, trace_alignments)[0] for c in length_convs]
                
                avg_goal = sum(goal_scores) / len(goal_scores) if goal_scores else 0
                avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
                std_goal = statistics.stdev(goal_scores) if len(goal_scores) > 1 else 0
                std_alignment = statistics.stdev(alignment_scores) if len(alignment_scores) > 1 else 0
                
                sequence_length_data.append({
                    'User_Personality': user_p,
                    'Environment_Personality': env_p,
                    'Sequence_Length': seq_length,
                    'Conversation_Count': len(length_convs),
                    'Mean_Goal_Achievement': round(avg_goal, 4),
                    'Std_Goal_Achievement': round(std_goal, 4),
                    'Mean_Alignment_Score': round(avg_alignment, 4),
                    'Std_Alignment_Score': round(std_alignment, 4),
                    'Min_Goal_Achievement': round(min(goal_scores) if goal_scores else 0, 4),
                    'Max_Goal_Achievement': round(max(goal_scores) if goal_scores else 0, 4),
                    'Min_Alignment_Score': round(min(alignment_scores) if alignment_scores else 0, 4),
                    'Max_Alignment_Score': round(max(alignment_scores) if alignment_scores else 0, 4)
                })
    
    if sequence_length_data:
        seq_df = pd.DataFrame(sequence_length_data)
        
        # Print detailed sequence length data
        print("\n" + "="*80)
        print("📊 SEQUENCE LENGTH DETAILED ANALYSIS")
        print("="*80)
        print(seq_df.to_string(index=False))
        
        # Create summary by sequence length
        length_summary = seq_df.groupby('Sequence_Length').agg({
            'Conversation_Count': 'sum',
            'Mean_Goal_Achievement': 'mean',
            'Mean_Alignment_Score': 'mean'
        }).round(4)
        length_summary.columns = ['Total_Conversations', 'Avg_Goal_Achievement', 'Avg_Alignment_Score']
        
        # Print sequence length summary
        print("\n" + "="*60)
        print("📏 SEQUENCE LENGTH SUMMARY")
        print("="*60)
        print(length_summary.to_string())
        
        # Create and print pivot tables for each sequence length
        unique_lengths = sorted(seq_df['Sequence_Length'].unique())
        
        for length in unique_lengths:
            length_data = seq_df[seq_df['Sequence_Length'] == length]
            
            # Goal achievement pivot
            goal_pivot = length_data.pivot_table(
                index='User_Personality',
                columns='Environment_Personality',
                values='Mean_Goal_Achievement',
                aggfunc='mean'
            )
            
            # Alignment score pivot
            alignment_pivot = length_data.pivot_table(
                index='User_Personality',
                columns='Environment_Personality',
                values='Mean_Alignment_Score',
                aggfunc='mean'
            )
            
            # Print length-specific pivots
            if not goal_pivot.empty:
                print(f"\n" + "="*50)
                print(f"🎯 SEQUENCE LENGTH {length} - GOAL ACHIEVEMENT")
                print("="*50)
                print(goal_pivot.round(4).to_string())
            
            if not alignment_pivot.empty:
                print(f"\n" + "="*50)
                print(f"📏 SEQUENCE LENGTH {length} - ALIGNMENT SCORE")
                print("="*50)
                print(alignment_pivot.round(4).to_string())
        
        # Print insights
        if len(unique_lengths) > 1:
            best_length_goal = length_summary['Avg_Goal_Achievement'].idxmax()
            worst_length_goal = length_summary['Avg_Goal_Achievement'].idxmin()
            best_length_alignment = length_summary['Avg_Alignment_Score'].idxmax()
            worst_length_alignment = length_summary['Avg_Alignment_Score'].idxmin()
            
            print(f"\n" + "="*60)
            print("💡 SEQUENCE LENGTH INSIGHTS")
            print("="*60)
            print(f"Sequence lengths analyzed: {list(unique_lengths)}")
            print()
            print("🏆 BEST PERFORMING LENGTHS:")
            print(f"  Goal Achievement: Length {best_length_goal} ({length_summary.loc[best_length_goal, 'Avg_Goal_Achievement']:.4f})")
            print(f"  Alignment Score:  Length {best_length_alignment} ({length_summary.loc[best_length_alignment, 'Avg_Alignment_Score']:.4f})")
            print()
            print("⚠️  WORST PERFORMING LENGTHS:")
            print(f"  Goal Achievement: Length {worst_length_goal} ({length_summary.loc[worst_length_goal, 'Avg_Goal_Achievement']:.4f})")
            print(f"  Alignment Score:  Length {worst_length_alignment} ({length_summary.loc[worst_length_alignment, 'Avg_Alignment_Score']:.4f})")
    
    else:
        print("⚠️ No sequence length data found for analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Generate personality analysis tables from conversation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_personality_analysis.py ./results
  python generate_personality_analysis.py /path/to/results
        """
    )
    
    parser.add_argument(
        "results_directory",
        help="Directory containing conversation data files (conversations.json, trace_alignments.json, etc.)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    results_dir = Path(args.results_directory)
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"❌ Error: Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    # Load data
    print(f"📂 Loading data from: {results_dir.absolute()}")
    conversations, trace_alignments, alignment_summary = load_files_from_folder(str(results_dir))
    
    if not conversations:
        print("❌ Error: No conversations loaded. Please check the results directory contains conversation data.")
        sys.exit(1)
    
    print(f"📊 Loaded {len(conversations)} conversations")
    
    # Generate analyses
    try:
        generate_user_vs_env_analysis(conversations, trace_alignments)
        generate_sequence_length_analysis(conversations, trace_alignments)
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 