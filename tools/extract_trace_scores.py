#!/usr/bin/env python3
"""
Extract Trace Scores - Generate LaTeX tables for trace alignment scores and goal achievement rates

This script processes experiment data and outputs LaTeX tables showing:
1. Goal Achievement rates by User Personality and Environment Archetype
2. Trace Alignment scores by User Personality and Environment Archetype  
3. Goal Achievement and Trace Alignment scores by Sequence Length and Environment Archetype
4. Goal Achievement and Trace Alignment scores by Sequence Length and User Archetype
5. Inner-Trace-Set Standard Deviation by Sequence Length

The script dynamically adapts to any sequence lengths found in the data.
Inner-trace-set standard deviation measures the variability within each trace_set_id group,
providing insights into consistency of agent performance across different instantiations.
"""

import json
import argparse
import sys
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# ANSI color codes for nice CLI output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_json_file(filepath: Path) -> Any:
    """Load JSON file safely with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Colors.FAIL}❌ File not found: {filepath}{Colors.ENDC}")
        return None
    except json.JSONDecodeError as e:
        print(f"{Colors.FAIL}❌ Invalid JSON in {filepath}: {e}{Colors.ENDC}")
        return None
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error loading {filepath}: {e}{Colors.ENDC}")
        return None

def load_experiment_data(data_dir: Path) -> Tuple[List[Dict], Dict[str, Any], List[Dict]]:
    """Load experiment data from directory."""
    conversations = []
    trace_alignments = {}
    alignment_summary = []
    
    # Try different possible filenames
    conversation_files = ["conversations.json", "conversation_data.json", "conversations_data.json"]
    alignment_files = ["trace_alignments.json", "alignments.json"]
    summary_files = ["alignment_summary.json", "summary.json", "trace_summary.json"]
    
    print(f"{Colors.OKBLUE}📂 Loading data from: {data_dir}{Colors.ENDC}")
    
    # Load conversations
    for filename in conversation_files:
        filepath = data_dir / filename
        if filepath.exists():
            conversations = load_json_file(filepath)
            if conversations:
                print(f"{Colors.OKGREEN}✅ Loaded {len(conversations)} conversations from {filename}{Colors.ENDC}")
                break
    
    # Load trace alignments
    for filename in alignment_files:
        filepath = data_dir / filename
        if filepath.exists():
            trace_alignments = load_json_file(filepath)
            if trace_alignments:
                print(f"{Colors.OKGREEN}✅ Loaded alignments for {len(trace_alignments)} trace sets from {filename}{Colors.ENDC}")
                break
    
    # Load alignment summary
    for filename in summary_files:
        filepath = data_dir / filename
        if filepath.exists():
            alignment_summary = load_json_file(filepath)
            if alignment_summary:
                print(f"{Colors.OKGREEN}✅ Loaded alignment summary for {len(alignment_summary)} trace sets from {filename}{Colors.ENDC}")
                break
    
    return conversations or [], trace_alignments or {}, alignment_summary or []

def get_similarity_score(conv: Dict[str, Any], trace_alignments: Dict[str, Any]) -> float:
    """Get similarity score from alignment data."""
    trace_set_id = conv.get("trace_set_id")
    instantiation_id = conv.get("instantiation_id", 0)
    
    if trace_set_id and trace_set_id in trace_alignments:
        alignment_data = trace_alignments[trace_set_id]
        alignments = alignment_data.get("alignments", [])
        
        if instantiation_id < len(alignments):
            alignment = alignments[instantiation_id]
            return alignment.get("similarity", 0.0)
    
    return 0.0

def get_goal_achievement_score(conv: Dict[str, Any], trace_alignments: Dict[str, Any]) -> float:
    """Get goal achievement score from alignment data."""
    trace_set_id = conv.get("trace_set_id")
    instantiation_id = conv.get("instantiation_id", 0)
    
    if trace_set_id and trace_set_id in trace_alignments:
        alignment_data = trace_alignments[trace_set_id]
        goal_results = alignment_data.get("goal_achievement_results", [])
        
        # Try to find by conversation_id first, then by instantiation_id
        for result in goal_results:
            if result.get("conversation_id", -1) == conv.get("conversation_id", -1):
                return result.get("score", 0.0)
        
        # Fallback to instantiation_id
        if instantiation_id < len(goal_results):
            return goal_results[instantiation_id].get("score", 0.0)
    
    return 0.0

def get_gt_sequence_size(conv: Dict[str, Any]) -> int:
    """Get ground truth tool sequence size."""
    user_source = conv.get('user_source', [])
    return len(user_source)

def calculate_stats(values: List[float]) -> Tuple[float, float, int]:
    """Calculate mean, std, and count from a list of values."""
    if not values:
        return 0.0, 0.0, 0
    
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    count = len(values)
    
    return mean_val, std_val, count

def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_stats_table(data: Dict[str, Dict[str, List[float]]], title: str):
    """Print a formatted statistics table."""
    print_section_header(title)
    
    if not data:
        print(f"{Colors.WARNING}No data available for {title.lower()}{Colors.ENDC}")
        return
    
    # Calculate column widths
    max_name_width = max(len(name) for name in data.keys()) + 2
    max_name_width = max(max_name_width, 20)
    
    # Print table header
    print(f"{Colors.BOLD}{'Category':<{max_name_width}} {'Count':<8} {'Trace Align':<15} {'Goal Achieve':<15}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'-'*max_name_width} {'-'*8} {'-'*15} {'-'*15}{Colors.ENDC}")
    
    # Sort categories for consistent output
    sorted_categories = sorted(data.keys())
    
    for category in sorted_categories:
        category_data = data[category]
        
        # Calculate stats for trace alignment
        trace_mean, trace_std, trace_count = calculate_stats(category_data.get('trace_alignment', []))
        
        # Calculate stats for goal achievement
        goal_mean, goal_std, goal_count = calculate_stats(category_data.get('goal_achievement', []))
        
        # Use trace_count as the primary count (should be same as goal_count)
        count = trace_count
        
        # Format the output
        trace_str = f"{trace_mean:.3f}±{trace_std:.3f}" if count > 0 else "N/A"
        goal_str = f"{goal_mean:.3f}±{goal_std:.3f}" if count > 0 else "N/A"
        
        print(f"{category:<{max_name_width}} {count:<8} {trace_str:<15} {goal_str:<15}")

def analyze_by_user_personality(conversations: List[Dict], trace_alignments: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Analyze scores by user personality."""
    data = defaultdict(lambda: defaultdict(list))
    
    for conv in conversations:
        user_personality = conv.get('user_personality_name', 'Unknown') or 'Unknown'
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        data[user_personality]['trace_alignment'].append(trace_score)
        data[user_personality]['goal_achievement'].append(goal_score)
    
    return dict(data)

def analyze_by_env_personality(conversations: List[Dict], trace_alignments: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Analyze scores by environment personality."""
    data = defaultdict(lambda: defaultdict(list))
    
    for conv in conversations:
        env_personality = conv.get('environment_personality_name', 'Unknown') or 'Unknown'
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        data[env_personality]['trace_alignment'].append(trace_score)
        data[env_personality]['goal_achievement'].append(goal_score)
    
    return dict(data)

def analyze_by_personality_combination(conversations: List[Dict], trace_alignments: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Analyze scores by user + environment personality combination."""
    data = defaultdict(lambda: defaultdict(list))
    
    for conv in conversations:
        user_personality = conv.get('user_personality_name', 'Unknown') or 'Unknown'
        env_personality = conv.get('environment_personality_name', 'Unknown') or 'Unknown'
        
        combination = f"{user_personality} + {env_personality}"
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        data[combination]['trace_alignment'].append(trace_score)
        data[combination]['goal_achievement'].append(goal_score)
    
    return dict(data)

def analyze_by_gt_sequence_size(conversations: List[Dict], trace_alignments: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Analyze scores by ground truth tool sequence size."""
    data = defaultdict(lambda: defaultdict(list))
    
    for conv in conversations:
        gt_size = get_gt_sequence_size(conv)
        size_category = f"Size {gt_size}"
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        data[size_category]['trace_alignment'].append(trace_score)
        data[size_category]['goal_achievement'].append(goal_score)
    
    return dict(data)

def calculate_inner_trace_set_std(conversations: List[Dict], trace_alignments: Dict) -> Tuple[float, float, int]:
    """Calculate average standard deviation within each trace_set_id group."""
    trace_set_groups = defaultdict(list)
    
    # Group conversations by trace_set_id
    for conv in conversations:
        trace_set_id = conv.get('trace_set_id')
        if trace_set_id:
            trace_score = get_similarity_score(conv, trace_alignments)
            goal_score = get_goal_achievement_score(conv, trace_alignments)
            trace_set_groups[trace_set_id].append({
                'trace_score': trace_score,
                'goal_score': goal_score
            })
    
    # Calculate std for each trace_set_id group
    trace_stds = []
    goal_stds = []
    valid_groups = 0
    
    for trace_set_id, group_data in trace_set_groups.items():
        if len(group_data) > 1:  # Need at least 2 points for std
            trace_scores = [item['trace_score'] for item in group_data]
            goal_scores = [item['goal_score'] for item in group_data]
            
            trace_std = statistics.stdev(trace_scores)
            goal_std = statistics.stdev(goal_scores)
            
            trace_stds.append(trace_std)
            goal_stds.append(goal_std)
            valid_groups += 1
    
    # Calculate average standard deviations
    avg_trace_std = statistics.mean(trace_stds) if trace_stds else 0.0
    avg_goal_std = statistics.mean(goal_stds) if goal_stds else 0.0
    
    return avg_trace_std, avg_goal_std, valid_groups

def analyze_inner_std_by_gt_sequence_size(conversations: List[Dict], trace_alignments: Dict) -> Dict[str, Dict[str, float]]:
    """Analyze inner-trace_set_id standard deviation by ground truth tool sequence size."""
    size_groups = defaultdict(lambda: defaultdict(list))
    
    # Group conversations by GT sequence size and trace_set_id
    for conv in conversations:
        gt_size = get_gt_sequence_size(conv)
        trace_set_id = conv.get('trace_set_id')
        
        if trace_set_id:
            trace_score = get_similarity_score(conv, trace_alignments)
            goal_score = get_goal_achievement_score(conv, trace_alignments)
            
            size_groups[gt_size][trace_set_id].append({
                'trace_score': trace_score,
                'goal_score': goal_score
            })
    
    # Calculate average std for each size category
    results = {}
    
    for gt_size, trace_set_data in size_groups.items():
        trace_stds = []
        goal_stds = []
        valid_groups = 0
        
        for trace_set_id, group_data in trace_set_data.items():
            if len(group_data) > 1:  # Need at least 2 points for std
                trace_scores = [item['trace_score'] for item in group_data]
                goal_scores = [item['goal_score'] for item in group_data]
                
                trace_std = statistics.stdev(trace_scores)
                goal_std = statistics.stdev(goal_scores)
                
                trace_stds.append(trace_std)
                goal_stds.append(goal_std)
                valid_groups += 1
        
        avg_trace_std = statistics.mean(trace_stds) if trace_stds else 0.0
        avg_goal_std = statistics.mean(goal_stds) if goal_stds else 0.0
        
        results[f"Size {gt_size}"] = {
            'trace_inner_std': avg_trace_std,
            'goal_inner_std': avg_goal_std,
            'valid_groups': valid_groups
        }
    
    return results

def print_inner_std_analysis(conversations: List[Dict], trace_alignments: Dict):
    """Print inner-trace_set_id standard deviation analysis."""
    print_section_header("INNER-TRACE_SET_ID STANDARD DEVIATION ANALYSIS")
    
    # Overall inner-std calculation
    avg_trace_std, avg_goal_std, valid_groups = calculate_inner_trace_set_std(conversations, trace_alignments)
    
    print(f"{Colors.OKGREEN}Overall Average Inner-Trace_Set_ID Std:{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Trace Alignment: {avg_trace_std:.4f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Goal Achievement: {avg_goal_std:.4f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Valid Groups (>1 sample): {valid_groups}{Colors.ENDC}")
    
    # Analysis by GT sequence size
    size_analysis = analyze_inner_std_by_gt_sequence_size(conversations, trace_alignments)
    
    print(f"\n{Colors.BOLD}Inner-Std by GT Tool Sequence Size:{Colors.ENDC}")
    print(f"{Colors.BOLD}{'Size':<12} {'Groups':<8} {'Trace Inner-Std':<18} {'Goal Inner-Std':<18}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'-'*12} {'-'*8} {'-'*18} {'-'*18}{Colors.ENDC}")
    
    # Sort by size for consistent output
    sorted_sizes = sorted(size_analysis.keys(), key=lambda x: int(x.split()[1]))
    
    for size_category in sorted_sizes:
        data = size_analysis[size_category]
        
        size_str = size_category.replace("Size ", "")
        groups = data['valid_groups']
        trace_std = data['trace_inner_std']
        goal_std = data['goal_inner_std']
        
        print(f"{size_str:<12} {groups:<8} {trace_std:<18.4f} {goal_std:<18.4f}")

def analyze_by_personality_environment_combination(conversations: List[Dict], trace_alignments: Dict) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Analyze scores by user personality and environment personality combination."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for conv in conversations:
        user_personality = conv.get('user_personality_name', 'Unknown') or 'Unknown'
        env_personality = conv.get('environment_personality_name', 'Unknown') or 'Unknown'
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        data[user_personality][env_personality]['trace_alignment'].append(trace_score)
        data[user_personality][env_personality]['goal_achievement'].append(goal_score)
    
    return dict(data)

def get_all_sequence_lengths(conversations: List[Dict]) -> List[int]:
    """Get all unique sequence lengths found in the data, sorted."""
    lengths = set()
    for conv in conversations:
        gt_size = get_gt_sequence_size(conv)
        lengths.add(gt_size)
    return sorted(list(lengths))

def analyze_by_length_environment_combination(conversations: List[Dict], trace_alignments: Dict) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """Analyze scores by sequence length and environment personality combination."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for conv in conversations:
        env_personality = conv.get('environment_personality_name', 'Unknown') or 'Unknown'
        gt_size = get_gt_sequence_size(conv)
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        data[gt_size][env_personality]['trace_alignment'].append(trace_score)
        data[gt_size][env_personality]['goal_achievement'].append(goal_score)
    
    return dict(data)

def analyze_by_length_user_combination(conversations: List[Dict], trace_alignments: Dict) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """Analyze scores by sequence length and user personality combination."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for conv in conversations:
        user_personality = conv.get('user_personality_name', 'Unknown') or 'Unknown'
        gt_size = get_gt_sequence_size(conv)
        
        trace_score = get_similarity_score(conv, trace_alignments)
        goal_score = get_goal_achievement_score(conv, trace_alignments)
        
        data[gt_size][user_personality]['trace_alignment'].append(trace_score)
        data[gt_size][user_personality]['goal_achievement'].append(goal_score)
    
    return dict(data)

def generate_latex_goal_achievement_table(conversations: List[Dict], trace_alignments: Dict) -> str:
    """Generate LaTeX table for Goal Achievement rates by User Personality and Environment Archetype."""
    data = analyze_by_personality_environment_combination(conversations, trace_alignments)
    
    # Define the order of personalities and environments as shown in the example
    personalities = ["Goal-Shifter", "Improviser", "Information Hider", "Planner", "Russian", "The Impatient User"]
    environments = ["Adversarial", "Buggy", "Perfect"]
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Goal Achievement rates by User Personality and Environment Archetype}")
    latex.append("  \\label{tab:goal_achievement}")
    latex.append("  \\begin{tabular}{|l|c|c|c|c|}")
    latex.append("    \\hline")
    latex.append("    \\textbf{User Personality} & \\textbf{Adversarial} & \\textbf{Buggy} & \\textbf{Perfect} & \\textbf{Average} \\\\ \\hline")
    
    # Calculate averages per environment
    env_totals = {env: [] for env in environments}
    
    for personality in personalities:
        row_data = []
        personality_scores = []
        
        for env in environments:
            if personality in data and env in data[personality]:
                scores = data[personality][env]['goal_achievement']
                if scores:
                    mean_score = statistics.mean(scores)
                    row_data.append(f"{mean_score:.3f}")
                    personality_scores.append(mean_score)
                    env_totals[env].extend(scores)
                else:
                    row_data.append("0.000")
                    personality_scores.append(0.0)
            else:
                row_data.append("0.000")
                personality_scores.append(0.0)
        
        # Calculate average for this personality
        avg_score = statistics.mean(personality_scores) if personality_scores else 0.0
        
        latex.append(f"    {personality} & {' & '.join(row_data)} & \\textbf{{{avg_score:.3f}}} \\\\ \\hline")
    
    # Calculate overall averages
    env_averages = []
    all_scores = []
    for env in environments:
        if env_totals[env]:
            env_avg = statistics.mean(env_totals[env])
            env_averages.append(f"{env_avg:.3f}")
            all_scores.extend(env_totals[env])
        else:
            env_averages.append("0.000")
    
    overall_avg = statistics.mean(all_scores) if all_scores else 0.0
    
    latex.append(f"    \\textbf{{Average}} & \\textbf{{{env_averages[0]}}} & \\textbf{{{env_averages[1]}}} & \\textbf{{{env_averages[2]}}} & \\textbf{{{overall_avg:.3f}}} \\\\ \\hline")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_latex_trace_alignment_table(conversations: List[Dict], trace_alignments: Dict) -> str:
    """Generate LaTeX table for Trace Alignment scores by User Personality and Environment Archetype."""
    data = analyze_by_personality_environment_combination(conversations, trace_alignments)
    
    # Define the order of personalities and environments as shown in the example
    personalities = ["Goal-Shifter", "Improviser", "Information Hider", "Planner", "Russian", "The Impatient User"]
    environments = ["Adversarial", "Buggy", "Perfect"]
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Trace Alignment scores by User Personality and Environment Archetype}")
    latex.append("  \\label{tab:trace_alignment}")
    latex.append("  \\begin{tabular}{|l|c|c|c|c|}")
    latex.append("    \\hline")
    latex.append("    \\textbf{User Personality} & \\textbf{Adversarial} & \\textbf{Buggy} & \\textbf{Perfect} & \\textbf{Average} \\\\ \\hline")
    
    # Calculate averages per environment
    env_totals = {env: [] for env in environments}
    
    for personality in personalities:
        row_data = []
        personality_scores = []
        
        for env in environments:
            if personality in data and env in data[personality]:
                scores = data[personality][env]['trace_alignment']
                if scores:
                    mean_score = statistics.mean(scores)
                    row_data.append(f"{mean_score:.3f}")
                    personality_scores.append(mean_score)
                    env_totals[env].extend(scores)
                else:
                    row_data.append("0.000")
                    personality_scores.append(0.0)
            else:
                row_data.append("0.000")
                personality_scores.append(0.0)
        
        # Calculate average for this personality
        avg_score = statistics.mean(personality_scores) if personality_scores else 0.0
        
        latex.append(f"    {personality:<18} & {' & '.join(row_data)} & \\textbf{{{avg_score:.3f}}} \\\\ \\hline")
    
    # Calculate overall averages
    env_averages = []
    all_scores = []
    for env in environments:
        if env_totals[env]:
            env_avg = statistics.mean(env_totals[env])
            env_averages.append(f"{env_avg:.3f}")
            all_scores.extend(env_totals[env])
        else:
            env_averages.append("0.000")
    
    overall_avg = statistics.mean(all_scores) if all_scores else 0.0
    
    latex.append(f"    \\textbf{{Average}}       & \\textbf{{{env_averages[0]}}} & \\textbf{{{env_averages[1]}}} & \\textbf{{{env_averages[2]}}} & \\textbf{{{overall_avg:.3f}}} \\\\ \\hline")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_latex_scores_by_length_environment_table(conversations: List[Dict], trace_alignments: Dict) -> str:
    """Generate LaTeX table for Goal Achievement and Trace Alignment scores by Sequence Length and Environment Archetype."""
    data = analyze_by_length_environment_combination(conversations, trace_alignments)
    
    # Get all available sequence lengths dynamically
    lengths = get_all_sequence_lengths(conversations)
    environments = ["Adversarial", "Buggy", "Perfect"]
    
    # Calculate number of columns needed (2 metrics * number of lengths)
    num_cols = len(lengths) * 2
    col_spec = "|l|" + "c|c|" * len(lengths)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Goal Achievement and Trace Alignment scores by Sequence Length and Environment Archetype}")
    latex.append("  \\label{tab:scores_by_length_environment}")
    latex.append("  \\resizebox{\\textwidth}{!}{%")
    latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    latex.append("      \\hline")
    
    # Build header rows
    header1_parts = ["\\multirow{2}{*}{\\textbf{Environment}}"]
    header2_parts = [""]
    
    for length in lengths:
        header1_parts.append(f"\\multicolumn{{2}}{{c|}}{{\\textbf{{Length {length}}}}}")
        header2_parts.extend(["\\textbf{Goal}", "\\textbf{Trace}"])
    
    latex.append(f"      {' & '.join(header1_parts)} \\\\")
    latex.append(f"      \\cline{{2-{num_cols + 1}}}")
    latex.append(f"      {' & '.join(header2_parts)} \\\\ \\hline")
    
    # Calculate averages for each length-environment combination
    length_env_totals = {length: {'goal': [], 'trace': []} for length in lengths}
    
    for env in environments:
        row_data = [env]
        
        for length in lengths:
            # Goal achievement score for this environment and length
            if length in data and env in data[length]:
                goal_scores = data[length][env]['goal_achievement']
                if goal_scores:
                    goal_mean = statistics.mean(goal_scores)
                    row_data.append(f"{goal_mean:.3f}")
                    length_env_totals[length]['goal'].extend(goal_scores)
                else:
                    row_data.append("0.000")
            else:
                row_data.append("0.000")
            
            # Trace alignment score for this environment and length
            if length in data and env in data[length]:
                trace_scores = data[length][env]['trace_alignment']
                if trace_scores:
                    trace_mean = statistics.mean(trace_scores)
                    row_data.append(f"{trace_mean:.3f}")
                    length_env_totals[length]['trace'].extend(trace_scores)
                else:
                    row_data.append("0.000")
            else:
                row_data.append("0.000")
        
        latex.append(f"      {' & '.join(row_data)} \\\\ \\hline")
    
    # Calculate and add average row
    avg_row_data = ["\\textbf{Average}"]
    for length in lengths:
        # Goal average for this length
        if length_env_totals[length]['goal']:
            goal_avg = statistics.mean(length_env_totals[length]['goal'])
            avg_row_data.append(f"\\textbf{{{goal_avg:.3f}}}")
        else:
            avg_row_data.append("\\textbf{0.000}")
        
        # Trace average for this length
        if length_env_totals[length]['trace']:
            trace_avg = statistics.mean(length_env_totals[length]['trace'])
            avg_row_data.append(f"\\textbf{{{trace_avg:.3f}}}")
        else:
            avg_row_data.append("\\textbf{0.000}")
    
    latex.append(f"      {' & '.join(avg_row_data)} \\\\ \\hline")
    latex.append("    \\end{tabular}%")
    latex.append("  }")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_latex_scores_by_length_user_table(conversations: List[Dict], trace_alignments: Dict) -> str:
    """Generate LaTeX table for Goal Achievement and Trace Alignment scores by Sequence Length and User Archetype."""
    data = analyze_by_length_user_combination(conversations, trace_alignments)
    
    # Get all available sequence lengths dynamically
    lengths = get_all_sequence_lengths(conversations)
    personalities = ["Goal-Shifter", "Improviser", "Information Hider", "Planner", "Russian", "The Impatient User"]
    
    # Calculate number of columns needed (2 metrics * number of lengths)
    num_cols = len(lengths) * 2
    col_spec = "|l|" + "c|c|" * len(lengths)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Goal Achievement and Trace Alignment scores by Sequence Length and User Archetype}")
    latex.append("  \\label{tab:scores_by_length_user}")
    latex.append("  \\resizebox{\\textwidth}{!}{%")
    latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    latex.append("      \\hline")
    
    # Build header rows
    header1_parts = ["\\multirow{2}{*}{\\textbf{User Personality}}"]
    header2_parts = [""]
    
    for length in lengths:
        header1_parts.append(f"\\multicolumn{{2}}{{c|}}{{\\textbf{{Length {length}}}}}")
        header2_parts.extend(["\\textbf{Goal}", "\\textbf{Trace}"])
    
    latex.append(f"      {' & '.join(header1_parts)} \\\\")
    latex.append(f"      \\cline{{2-{num_cols + 1}}}")
    latex.append(f"      {' & '.join(header2_parts)} \\\\ \\hline")
    
    # Calculate averages for each length-personality combination
    length_personality_totals = {length: {'goal': [], 'trace': []} for length in lengths}
    
    for personality in personalities:
        row_data = [personality]
        
        for length in lengths:
            # Goal achievement score for this personality and length
            if length in data and personality in data[length]:
                goal_scores = data[length][personality]['goal_achievement']
                if goal_scores:
                    goal_mean = statistics.mean(goal_scores)
                    row_data.append(f"{goal_mean:.3f}")
                    length_personality_totals[length]['goal'].extend(goal_scores)
                else:
                    row_data.append("0.000")
            else:
                row_data.append("0.000")
            
            # Trace alignment score for this personality and length
            if length in data and personality in data[length]:
                trace_scores = data[length][personality]['trace_alignment']
                if trace_scores:
                    trace_mean = statistics.mean(trace_scores)
                    row_data.append(f"{trace_mean:.3f}")
                    length_personality_totals[length]['trace'].extend(trace_scores)
                else:
                    row_data.append("0.000")
            else:
                row_data.append("0.000")
        
        latex.append(f"      {' & '.join(row_data)} \\\\ \\hline")
    
    # Calculate and add average row
    avg_row_data = ["\\textbf{Average}"]
    for length in lengths:
        # Goal average for this length
        if length_personality_totals[length]['goal']:
            goal_avg = statistics.mean(length_personality_totals[length]['goal'])
            avg_row_data.append(f"\\textbf{{{goal_avg:.3f}}}")
        else:
            avg_row_data.append("\\textbf{0.000}")
        
        # Trace average for this length
        if length_personality_totals[length]['trace']:
            trace_avg = statistics.mean(length_personality_totals[length]['trace'])
            avg_row_data.append(f"\\textbf{{{trace_avg:.3f}}}")
        else:
            avg_row_data.append("\\textbf{0.000}")
    
    latex.append(f"      {' & '.join(avg_row_data)} \\\\ \\hline")
    latex.append("    \\end{tabular}%")
    latex.append("  }")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def generate_latex_inner_std_by_length_table(conversations: List[Dict], trace_alignments: Dict) -> str:
    """Generate LaTeX table for Inner-Trace-Set Standard Deviation by Sequence Length."""
    size_analysis = analyze_inner_std_by_gt_sequence_size(conversations, trace_alignments)
    
    # Get all available sequence lengths dynamically
    lengths = get_all_sequence_lengths(conversations)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Inner-Trace-Set Standard Deviation by Sequence Length}")
    latex.append("  \\label{tab:inner_std_by_length}")
    latex.append("  \\begin{tabular}{|c|c|c|c|}")
    latex.append("    \\hline")
    latex.append("    \\textbf{Sequence Length} & \\textbf{Valid Groups} & \\textbf{Trace Inner-Std} & \\textbf{Goal Inner-Std} \\\\ \\hline")
    
    # Sort lengths for consistent output
    sorted_lengths = sorted(lengths)
    
    for length in sorted_lengths:
        size_key = f"Size {length}"
        if size_key in size_analysis:
            data = size_analysis[size_key]
            groups = data['valid_groups']
            trace_std = data['trace_inner_std']
            goal_std = data['goal_inner_std']
            
            latex.append(f"    {length} & {groups} & {trace_std:.4f} & {goal_std:.4f} \\\\ \\hline")
        else:
            latex.append(f"    {length} & 0 & 0.0000 & 0.0000 \\\\ \\hline")
    
    # Calculate and add overall average
    avg_trace_std, avg_goal_std, total_valid_groups = calculate_inner_trace_set_std(conversations, trace_alignments)
    latex.append(f"    \\textbf{{Overall}} & \\textbf{{{total_valid_groups}}} & \\textbf{{{avg_trace_std:.4f}}} & \\textbf{{{avg_goal_std:.4f}}} \\\\ \\hline")
    
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def print_summary_statistics(conversations: List[Dict], trace_alignments: Dict):
    """Print overall summary statistics."""
    print_section_header("OVERALL SUMMARY")
    
    total_conversations = len(conversations)
    total_trace_sets = len(set(conv.get('trace_set_id') for conv in conversations if conv.get('trace_set_id')))
    
    # Calculate overall stats
    all_trace_scores = [get_similarity_score(conv, trace_alignments) for conv in conversations]
    all_goal_scores = [get_goal_achievement_score(conv, trace_alignments) for conv in conversations]
    
    trace_mean, trace_std, _ = calculate_stats(all_trace_scores)
    goal_mean, goal_std, _ = calculate_stats(all_goal_scores)
    
    print(f"{Colors.OKGREEN}Total Conversations: {total_conversations}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Total Trace Sets: {total_trace_sets}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Overall Trace Alignment: {trace_mean:.3f}±{trace_std:.3f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Overall Goal Achievement: {goal_mean:.3f}±{goal_std:.3f}{Colors.ENDC}")
    
    # Personality distribution
    user_personalities = set(conv.get('user_personality_name', 'Unknown') or 'Unknown' for conv in conversations)
    env_personalities = set(conv.get('environment_personality_name', 'Unknown') or 'Unknown' for conv in conversations)
    
    print(f"{Colors.OKCYAN}User Personalities: {len(user_personalities)} ({', '.join(sorted(user_personalities))}){Colors.ENDC}")
    print(f"{Colors.OKCYAN}Environment Personalities: {len(env_personalities)} ({', '.join(sorted(env_personalities))}){Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for trace alignment scores and goal achievement rates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/experiment/results
  %(prog)s /path/to/experiment/results --no-color
  %(prog)s /path/to/experiment/results --output-file latex_tables.tex
        """
    )
    
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing experiment data files (conversations.json, trace_alignments.json, etc.)"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Save LaTeX tables to file instead of printing to console"
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Check if data directory exists
    if not args.data_dir.exists():
        print(f"{Colors.FAIL}❌ Data directory does not exist: {args.data_dir}{Colors.ENDC}")
        sys.exit(1)
    
    if not args.data_dir.is_dir():
        print(f"{Colors.FAIL}❌ Path is not a directory: {args.data_dir}{Colors.ENDC}")
        sys.exit(1)
    
    # Load experiment data
    conversations, trace_alignments, alignment_summary = load_experiment_data(args.data_dir)
    
    if not conversations:
        print(f"{Colors.FAIL}❌ No conversations loaded. Please check your data directory.{Colors.ENDC}")
        sys.exit(1)
    
    if not trace_alignments:
        print(f"{Colors.WARNING}⚠️ No trace alignments loaded. Some analyses may be incomplete.{Colors.ENDC}")
    
    # Redirect output if requested
    if args.output_file:
        sys.stdout = open(args.output_file, 'w', encoding='utf-8')
        print(f"% Extract Trace Scores - Analysis Results")
        print(f"% Data Directory: {args.data_dir}")
        print(f"% Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    try:
        # Generate LaTeX tables
        print("% Goal Achievement rates by User Personality and Environment Archetype")
        goal_achievement_table = generate_latex_goal_achievement_table(conversations, trace_alignments)
        print(goal_achievement_table)
        print()
        
        print("% Trace Alignment scores by User Personality and Environment Archetype")
        trace_alignment_table = generate_latex_trace_alignment_table(conversations, trace_alignments)
        print(trace_alignment_table)
        print()
        
        print("% Goal Achievement and Trace Alignment scores by Sequence Length and Environment Archetype")
        scores_by_length_environment_table = generate_latex_scores_by_length_environment_table(conversations, trace_alignments)
        print(scores_by_length_environment_table)
        print()
        
        print("% Goal Achievement and Trace Alignment scores by Sequence Length and User Archetype")
        scores_by_length_user_table = generate_latex_scores_by_length_user_table(conversations, trace_alignments)
        print(scores_by_length_user_table)
        print()
        
        print("% Inner-Trace-Set Standard Deviation by Sequence Length")
        inner_std_by_length_table = generate_latex_inner_std_by_length_table(conversations, trace_alignments)
        print(inner_std_by_length_table)
        
        if not args.output_file:
            print(f"\n{Colors.OKGREEN}✅ LaTeX tables generated successfully!{Colors.ENDC}")
            
            # Also print the inner-std analysis to console for debugging/verification
            print_inner_std_analysis(conversations, trace_alignments)
        
        if args.output_file:
            # Print to original stdout (console) not the redirected file
            original_stdout = sys.__stdout__
            print(f"{Colors.OKGREEN}📁 LaTeX tables saved to: {args.output_file}{Colors.ENDC}", file=original_stdout)
    
    finally:
        # Restore stdout if we redirected it
        if args.output_file:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main() 