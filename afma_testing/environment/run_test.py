#!/usr/bin/env python3
"""
Script to run concurrent environment simulation tests and log all results.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.text import Text
from litellm.caching.caching import Cache
import litellm
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

litellm.cache = Cache(type="disk")

from environment.concurrent_test_runner import ConcurrentEnvironmentTestRunner


def create_similarity_table(results_by_config):
    """Create a rich table showing similarity scores by file count and task complexity."""
    console = Console()
    
    # Extract file counts and task complexities
    file_counts = sorted(set(config['file_count'] for config in results_by_config.values()))
    task_complexities = sorted(set(config['task_complexity'] for config in results_by_config.values()))
    
    # Create table
    table = Table(title="Environment Simulation Results: Average ± Std Similarity Scores")
    
    # Add columns
    table.add_column("Files", style="cyan", no_wrap=True)
    for complexity in task_complexities:
        table.add_column(f"T{complexity}", style="magenta", justify="center")
    
    # Add rows
    for file_count in file_counts:
        row = [str(file_count)]
        for task_complexity in task_complexities:
            # Find the configuration
            config_key = f"{file_count}_files_{task_complexity}_complexity"
            if config_key in results_by_config:
                config = results_by_config[config_key]
                if config['successful_runs'] > 0:
                    avg_similarity = config.get('average_similarity', 0.0)
                    std_similarity = config.get('std_similarity', 0.0)
                    
                    # Format the display text
                    if config['successful_runs'] == 1:
                        # Single run, no std dev
                        display_text = f"{avg_similarity:.3f}"
                    else:
                        # Multiple runs, show avg ± std
                        display_text = f"{avg_similarity:.3f}±{std_similarity:.3f}"
                    
                    # Apply color coding based on average
                    if avg_similarity >= 0.95:
                        # Green for excellent similarity
                        row.append(f"[green]{display_text}[/green]")
                    elif avg_similarity >= 0.85:
                        # Yellow for good similarity
                        row.append(f"[yellow]{display_text}[/yellow]")
                    else:
                        # Red for poor similarity
                        row.append(f"[red]{display_text}[/red]")
                else:
                    # Red for failed tests
                    row.append("[red]FAIL[/red]")
            else:
                row.append("-")
        table.add_row(*row)
    
    console.print(table)
    print()


def plot_similarity_histogram(test_results, results_dir, timestamp, filter_outliers=True, threshold=0.3):
    """Plot histogram of all similarity scores and save to file."""
    
    # Extract all similarity scores
    all_similarity_scores = []
    for result in test_results:
        if result.get('success', False):
            similarity_score = result.get('comparison_result', {}).get('similarity_score', 0.0)
            all_similarity_scores.append(similarity_score)
    
    if not all_similarity_scores:
        print("❌ No similarity scores found for histogram")
        return
    
    # Filter outliers if requested
    if filter_outliers:
        similarity_scores = [score for score in all_similarity_scores if score >= threshold]
        removed_count = len(all_similarity_scores) - len(similarity_scores)
        print(f"📊 Plotting histogram of {len(similarity_scores)} similarity scores (filtered, removed {removed_count} outliers)...")
    else:
        similarity_scores = all_similarity_scores
        removed_count = 0
        print(f"📊 Plotting histogram of {len(similarity_scores)} similarity scores (unfiltered)...")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    n, bins, patches = plt.hist(similarity_scores, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics lines
    mean_score = np.mean(similarity_scores)
    median_score = np.median(similarity_scores)
    
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
    plt.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
    
    # Color code the bars based on similarity thresholds
    for i, (patch, bin_edge) in enumerate(zip(patches, bins[:-1])):
        if bin_edge >= 0.95:
            patch.set_facecolor('lightgreen')
        elif bin_edge >= 0.85:
            patch.set_facecolor('lightyellow')
        else:
            patch.set_facecolor('lightcoral')
    
    # Customize the plot
    title = 'Distribution of Similarity Scores\nEnvironment Simulation Test Results'
    if filter_outliers and removed_count > 0:
        title += f' (Filtered, {removed_count} outliers removed)'
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Similarity Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add text box with statistics
    if filter_outliers and removed_count > 0:
        stats_text = (f'Filtered Results:\n'
                      f'Tests: {len(similarity_scores)}\n'
                      f'Removed: {removed_count}\n'
                      f'Mean: {mean_score:.3f}\n'
                      f'Std Dev: {np.std(similarity_scores):.3f}\n'
                      f'Min: {min(similarity_scores):.3f}\n'
                      f'Max: {max(similarity_scores):.3f}\n'
                      f'25th percentile: {np.percentile(similarity_scores, 25):.3f}\n'
                      f'75th percentile: {np.percentile(similarity_scores, 75):.3f}')
    else:
        stats_text = (f'Total Tests: {len(similarity_scores)}\n'
                      f'Mean: {mean_score:.3f}\n'
                      f'Std Dev: {np.std(similarity_scores):.3f}\n'
                      f'Min: {min(similarity_scores):.3f}\n'
                      f'Max: {max(similarity_scores):.3f}\n'
                      f'25th percentile: {np.percentile(similarity_scores, 25):.3f}\n'
                      f'75th percentile: {np.percentile(similarity_scores, 75):.3f}')
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add color legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.7, label='Excellent (≥0.95)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightyellow', alpha=0.7, label='Good (≥0.85)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.7, label='Poor (<0.85)')
    ]
    plt.gca().add_artist(plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.85)))
    
    plt.tight_layout()
    
    # Save the plot
    histogram_file = results_dir / f"similarity_histogram_{timestamp}.png"
    plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
    print(f"📈 Histogram saved to: {histogram_file}")
    
    # Also save as PDF for publication quality
    pdf_file = results_dir / f"similarity_histogram_{timestamp}.pdf"
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    print(f"📊 PDF version saved to: {pdf_file}")
    
    plt.close()  # Close the figure to free memory
    
    return similarity_scores, removed_count


async def main():
    """Run concurrent environment tests and save results."""
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Create results directory
    results_dir = Path("results/environment_testing")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"concurrent_test_run_{timestamp}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    logger.info("Starting concurrent environment simulation testing")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Initialize concurrent test runner
        runner = ConcurrentEnvironmentTestRunner()
        
        logger.info("Running grid tests with all configurations")
        
        # Run the grid tests
        results = await runner.run_grid_tests()
        
        # Save detailed results
        result_file = results_dir / f"concurrent_test_results_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {result_file}")
        
        # Print summary
        summary = results["summary"]
        print("\n" + "="*80)
        print("CONCURRENT ENVIRONMENT SIMULATION TEST RESULTS")
        print("="*80)
        
        print(f"📊 Overall Results:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Successful tests: {summary['successful_tests']}")
        print(f"   Failed tests: {summary['failed_tests']}")
        print(f"   Success rate: {summary['success_rate']:.2%}")
        print(f"   Average execution time: {summary['average_execution_time']:.2f}s")
        
        if summary.get("similarity_scores"):
            print(f"\n🎯 Similarity Scores:")
            print(f"   Average: {summary['similarity_scores']['average']:.3f}")
            print(f"   Min: {summary['similarity_scores']['min']:.3f}")
            print(f"   Max: {summary['similarity_scores']['max']:.3f}")
        
        print(f"\n🔥 Detailed Results Matrix:")
        print("Legend: Files = Initial State Complexity, T1-T10 = Task Complexity Levels")
        print("Color coding: [green]≥0.95[/green] [yellow]≥0.85[/yellow] [red]<0.85[/red]")
        print()
        
        # Create and display the similarity table
        create_similarity_table(summary["results_by_configuration"])
        
        # Plot similarity histogram
        filtered_scores, removed_outliers = plot_similarity_histogram(results["test_results"], results_dir, timestamp)
        
        # Show outlier filtering summary
        if removed_outliers > 0:
            print(f"🔍 Outlier Analysis:")
            print(f"   Removed {removed_outliers} outliers (scores < 0.3)")
            print(f"   Filtered mean: {np.mean(filtered_scores):.3f}")
            print(f"   Filtered std: {np.std(filtered_scores):.3f}")
            print(f"   Outlier rate: {removed_outliers/summary['successful_tests']*100:.1f}%")
        
        # Show configuration analysis
        print("📈 Configuration Analysis:")
        excellent_configs = []
        good_configs = []
        poor_configs = []
        
        for config_key, config_summary in summary["results_by_configuration"].items():
            if config_summary['successful_runs'] > 0:
                similarity = config_summary.get('average_similarity', 0.0)
                config_desc = f"{config_summary['file_count']} files, {config_summary['task_complexity']} complexity"
                
                if similarity >= 0.95:
                    excellent_configs.append((config_desc, similarity))
                elif similarity >= 0.85:
                    good_configs.append((config_desc, similarity))
                else:
                    poor_configs.append((config_desc, similarity))
        
        print(f"   🟢 Excellent (≥0.95): {len(excellent_configs)} configurations")
        print(f"   🟡 Good (≥0.85): {len(good_configs)} configurations")
        print(f"   🔴 Poor (<0.85): {len(poor_configs)} configurations")
        
        if poor_configs:
            print(f"\n   Poor performing configurations:")
            for config_desc, similarity in sorted(poor_configs, key=lambda x: x[1]):
                print(f"     • {config_desc}: {similarity:.3f}")
        
        # Show some example results
        if results["test_results"]:
            print(f"\n🔍 Sample Test Results:")
            successful_tests = [r for r in results["test_results"] if r.get("success", False)]
            if successful_tests:
                sample = successful_tests[0]
                print(f"   Session {sample['session_id']}: {sample['file_count']} files, {sample['task_complexity']} complexity")
                print(f"     Similarity: {sample['comparison_result'].get('similarity_score', 0.0):.3f}")
                print(f"     Task: {sample['task_description'][:100]}...")
        
        print("\n" + "="*80)
        print(f"📁 Results saved to: {result_file}")
        print(f"📝 Log saved to: {log_file}")
        print("="*80)
        
    except Exception as e:
        logger.exception(f"Concurrent test script failed: {e}")
        print(f"❌ Concurrent test script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 