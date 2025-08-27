#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path
from loguru import logger
import litellm
from litellm.caching import Cache

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from afma_testing.personality_adherence.personality_adherence_tester import PersonalityAdherenceTester

litellm.cache = Cache(type="disk", cache_dir="~/.litellm_cache")

async def main():
    """Main function to run personality adherence testing."""
    if len(sys.argv) != 2:
        print("Usage: python run_personality_adherence_test.py <config_path>")
        print("Example: python afma_testing/personality_adherence/run_personality_adherence_test.py afma_testing/personality_adherence/config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Create tester and run test
        tester = PersonalityAdherenceTester(config_path)
        summary = await tester.run_test()
        
        logger.success("Personality adherence testing completed successfully!")
        # Pretty print summary stats
        print("\n--- Personality Adherence Test Summary ---")
        print(f"Total conversations tested: {summary.total_conversations}")
        print(f"Average User Personality Score: {summary.avg_user_personality_score:.2f} (StdDev: {summary.std_user_personality_score:.2f})")
        print(f"Average Environment Personality Score: {summary.avg_environment_personality_score:.2f} (StdDev: {summary.std_environment_personality_score:.2f})")
        
        print("\n--- Scores by User Personality ---")
        if summary.results_by_user_personality:
            for name, data in summary.results_by_user_personality.items():
                print(f"  - {name} ({data['count']} conversations): Avg Score: {data['avg_score']:.2f}, StdDev: {data['std_dev']:.2f}")
        else:
            print("  No user personality data available.")

        print("\n--- Scores by Environment Personality ---")
        if summary.results_by_environment_personality:
            for name, data in summary.results_by_environment_personality.items():
                print(f"  - {name} ({data['count']} conversations): Avg Score: {data['avg_score']:.2f}, StdDev: {data['std_dev']:.2f}")
        else:
            print("  No environment personality data available.")

        print("----------------------------------------")
        
        # Display information about generated visualizations
        output_dir = Path(tester.config.get("output_dir", "results/personality_adherence"))
        radar_chart = output_dir / "personality_adherence_radar_chart.png"
        latex_table = output_dir / "personality_adherence_table.tex"
        
        print("\n--- Generated Visualizations ---")
        if radar_chart.exists():
            print(f"  - Circular radar chart: {radar_chart}")
        if latex_table.exists():
            print(f"  - LaTeX table: {latex_table}")
        print("----------------------------------------")

        return 0
        
    except Exception as e:
        logger.exception(f"Error during personality adherence testing: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 