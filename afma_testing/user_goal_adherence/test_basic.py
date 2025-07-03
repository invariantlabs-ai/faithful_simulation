#!/usr/bin/env python3

import asyncio
import json
import sys
from pathlib import Path
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from afma_testing.user_goal_adherence.user_goal_adherence_assessor import UserGoalAdherenceAssessor

async def test_basic_assessment():
    """Test basic functionality of the assessor with a simple example."""
    
    # Create a simple test conversation
    test_conversation = {
        "user_goal": "I want to find a Python chatbot repository on GitHub and fork it.",
        "history": [
            {
                "role": "user",
                "content": "I want to find a Python chatbot repository on GitHub and fork it."
            },
            {
                "role": "assistant", 
                "content": "I'll help you find a Python chatbot repository on GitHub and fork it for you."
            },
            {
                "role": "user",
                "content": "Great! Can you also help me set up a development environment?"
            }
        ]
    }
    
    # Test configuration
    llm_config = {
        "model": "gpt-4.1-mini",
        "temperature": 0.1,
        "top_p": 0.9,
        "timeout": 30
    }
    
    # Create assessor
    assessor = UserGoalAdherenceAssessor(llm_config, concurrency=1)
    
    # Test assessment
    logger.info("Testing basic assessment functionality...")
    
    try:
        results = await assessor.assess_conversation(0, test_conversation)
        
        logger.info(f"Assessment completed. Got {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"  Message {i+1}: Score={result.score}, Reasoning={result.reasoning[:100]}...")
        
        # Check that we got results for both user messages
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        # The first message should score 1 (adheres to goal)
        # The second message should score 0 (asks for additional help beyond goal)
        logger.info("Basic test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

async def main():
    """Run basic tests."""
    logger.info("Running basic tests for user goal adherence module...")
    
    success = await test_basic_assessment()
    
    if success:
        logger.success("All basic tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 