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
from litellm.caching.caching import Cache
import litellm

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

litellm.cache = Cache(type="disk")

from src.afma.environment_testing.concurrent_test_runner import ConcurrentEnvironmentTestRunner


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
        
        print(f"\n📈 Results by Configuration:")
        for config_key, config_summary in summary["results_by_configuration"].items():
            file_count = config_summary['file_count']
            task_complexity = config_summary['task_complexity']
            success_rate = config_summary['successful_runs'] / config_summary['total_runs']
            
            print(f"   {file_count} files, {task_complexity} complexity: {config_summary['successful_runs']}/{config_summary['total_runs']} successful ({success_rate:.1%})")
            if config_summary.get("average_similarity"):
                print(f"     Average similarity: {config_summary['average_similarity']:.3f}")
        
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