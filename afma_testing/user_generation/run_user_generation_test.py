#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path
from loguru import logger
import litellm
from litellm.caching import Cache

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from afma_testing.user_generation.user_generation_tester import UserGenerationTester

litellm.cache = Cache(type="disk")

async def main():
    """Main function to run user generation testing."""
    if len(sys.argv) != 2:
        print("Usage: python run_user_generation_test.py <config_path>")
        print("Example: python run_user_generation_test.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Create tester and run test
        tester = UserGenerationTester(config_path)
        summary = await tester.run_test()
        
        logger.success("User generation testing completed successfully!")
        return 0
        
    except Exception as e:
        logger.exception(f"Error during user generation testing: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 