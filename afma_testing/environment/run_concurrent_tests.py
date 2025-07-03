"""
Script to run concurrent environment testing with grid parameters.
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from afma.environment_testing.concurrent_test_runner import ConcurrentEnvironmentTestRunner


async def main():
    """Main function to run concurrent environment tests."""
    try:
        logger.info("Starting concurrent environment testing")
        
        # Initialize the concurrent test runner
        runner = ConcurrentEnvironmentTestRunner()
        
        # Run grid tests
        results = await runner.run_grid_tests()
        
        # Print summary
        summary = results["summary"]
        logger.success(f"Grid testing completed!")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Successful tests: {summary['successful_tests']}")
        logger.info(f"Failed tests: {summary['failed_tests']}")
        logger.info(f"Success rate: {summary['success_rate']:.2%}")
        
        if summary.get("similarity_scores"):
            logger.info(f"Average similarity: {summary['similarity_scores']['average']:.3f}")
            logger.info(f"Min similarity: {summary['similarity_scores']['min']:.3f}")
            logger.info(f"Max similarity: {summary['similarity_scores']['max']:.3f}")
        
        logger.info(f"Results saved to: {results['results_file']}")
        
        # Print configuration-specific results
        logger.info("\nResults by configuration:")
        for config_key, config_summary in summary["results_by_configuration"].items():
            logger.info(f"  {config_key}: {config_summary['successful_runs']}/{config_summary['total_runs']} successful")
            if config_summary.get("average_similarity"):
                logger.info(f"    Average similarity: {config_summary['average_similarity']:.3f}")
        
        return results
        
    except Exception as e:
        logger.exception(f"Concurrent testing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 