"""
Main test runner that orchestrates the complete environment testing pipeline.
"""

import json
import os
import asyncio
from typing import Dict, Any, List
from loguru import logger
import yaml

from .environment_wrapper import TestableEnvironment, TestableSimulatedEnvironment
from .state_generator import EnvironmentStateGenerator
from .task_generator import TaskGenerator
from .state_comparator import StateComparator


class EnvironmentTestRunner:
    """Main test runner for environment simulation testing."""
    
    def __init__(self, config_path: str = "src/afma/environment_testing/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.test_config = self.config.get("environment_testing", {})
        self.llm_config = self.test_config.get("litellm", {})
        
        # Initialize components
        self.state_generator = EnvironmentStateGenerator(self.llm_config, self.test_config)
        self.task_generator = TaskGenerator(self.llm_config, self.test_config)
        self.state_comparator = StateComparator(self.llm_config, self.test_config)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def run_single_test(self, difficulty: int, mcp_config_path: str = "agent_mcp_configs/files.json") -> Dict[str, Any]:
        """
        Run a single environment test with the specified difficulty.
        
        Args:
            difficulty: Number of files to create in the environment
            mcp_config_path: Path to the MCP configuration file
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Starting environment test with difficulty {difficulty}")
        
        # Initialize environments
        real_env = TestableEnvironment(mcp_config_path, timeout=10, llm_config=self.llm_config)
        simulated_env = TestableSimulatedEnvironment(mcp_config_path, self.llm_config, timeout=10, session_dir=real_env.session_dir)
        
        test_result = {
            "difficulty": difficulty,
            "mcp_config_path": mcp_config_path,
            "success": False,
            "error": None,
            "initial_state": {},
            "task_description": "",
            "tool_sequence": [],
            "real_final_state": {},
            "simulated_final_state": {},
            "comparison_result": {},
            "execution_log": []
        }
        
        try:
            # Step 1: Generate and create initial environment state
            logger.info("Step 1: Generating initial environment state")
            initial_state = await self.state_generator.generate_environment_state(difficulty, real_env)
            if not initial_state:
                raise Exception("Failed to generate initial environment state")
            
            test_result["initial_state"] = initial_state
            test_result["execution_log"].append("Initial environment state created successfully")
            
            # Step 2: Get initial state from real environment
            logger.info("Step 2: Getting initial state from real environment")
            real_initial_state = await real_env.get_state()
            test_result["execution_log"].append(f"Retrieved initial state with {len(real_initial_state)} files")
            
            # Step 2.5: Initialize simulated environment with the same initial state
            logger.info("Step 2.5: Initializing simulated environment with initial state")
            await self._initialize_simulated_environment(simulated_env, real_initial_state)
            test_result["execution_log"].append(f"Initialized simulated environment with {len(real_initial_state)} files")
            
            # Step 3: Generate task and tool sequence
            logger.info("Step 3: Generating task and tool sequence")
            
            # Get available tools from the real environment
            await real_env.collect_resources()
            available_tools = real_env.tools if real_env.tools else []
            
            task_description, tool_sequence = await self.task_generator.generate_task_and_sequence(
                real_initial_state, 
                available_tools,
                max_complexity=5
            )
            
            test_result["task_description"] = task_description
            test_result["tool_sequence"] = tool_sequence
            
            logger.info(f"Generated task: {task_description}")
            logger.info(f"Generated {len(tool_sequence)} tool calls")
            
            # Step 4: Execute tool sequence in both environments
            logger.info("Step 4: Executing tool sequence in both environments")
            
            # Execute in real environment
            real_execution_results = []
            for i, tool_call in enumerate(tool_sequence):
                try:
                    result = await real_env.call_tool(
                        tool_call["tool_name"],
                        json.dumps(tool_call["arguments"]),
                        f"real_call_{i}"
                    )
                    real_execution_results.append({
                        "tool_call": tool_call,
                        "result": result[1],
                        "success": not result[1].startswith("Error")
                    })
                    logger.debug(f"Real environment tool call {i+1}/{len(tool_sequence)} completed")
                except Exception as e:
                    logger.error(f"Real environment tool call {i+1} failed: {e}")
                    real_execution_results.append({
                        "tool_call": tool_call,
                        "result": str(e),
                        "success": False
                    })
            
            # Execute in simulated environment
            simulated_execution_results = []
            for i, tool_call in enumerate(tool_sequence):
                try:
                    result = await simulated_env.call_tool(
                        tool_call["tool_name"],
                        json.dumps(tool_call["arguments"]),
                        f"simulated_call_{i}"
                    )
                    simulated_execution_results.append({
                        "tool_call": tool_call,
                        "result": result[1],
                        "success": not result[1].startswith("Error")
                    })
                    logger.debug(f"Simulated environment tool call {i+1}/{len(tool_sequence)} completed")
                except Exception as e:
                    logger.error(f"Simulated environment tool call {i+1} failed: {e}")
                    simulated_execution_results.append({
                        "tool_call": tool_call,
                        "result": str(e),
                        "success": False
                    })
            
            test_result["real_execution_results"] = real_execution_results
            test_result["simulated_execution_results"] = simulated_execution_results
            test_result["execution_log"].append("Tool sequence executed in both environments")
            
            # Step 5: Get final states from both environments
            logger.info("Step 5: Getting final states from both environments")
            real_final_state = await real_env.get_state()
            simulated_final_state = await simulated_env.get_state()
            
            test_result["real_final_state"] = real_final_state
            test_result["simulated_final_state"] = simulated_final_state
            test_result["execution_log"].append(f"Retrieved final states: real={len(real_final_state)} files, simulated={len(simulated_final_state)} files")
            
            # Step 6: Compare states
            logger.info("Step 6: Comparing environment states")
            comparison_result = await self.state_comparator.compare_states(
                real_final_state, simulated_final_state, task_description
            )
            
            test_result["comparison_result"] = comparison_result
            test_result["execution_log"].append(f"Comparison completed with similarity score: {comparison_result.get('similarity_score', 0.0):.3f}")
            
            # Step 7: Cleanup (if enabled)
            if self.test_config.get("test", {}).get("cleanup_after_test", True):
                logger.info("Step 7: Cleaning up test environment")
                await self.state_generator.cleanup_environment(real_env)
                test_result["execution_log"].append("Environment cleanup completed")
            
            test_result["success"] = True
            logger.success(f"Environment test completed successfully with similarity score: {comparison_result.get('similarity_score', 0.0):.3f}")
            
        except Exception as e:
            logger.exception(f"Environment test failed: {e}")
            test_result["error"] = str(e)
            test_result["execution_log"].append(f"Test failed with error: {e}")
        
        return test_result
    
    async def run_multiple_tests(self, difficulties: List[int], mcp_config_path: str = "agent_mcp_configs/files.json") -> Dict[str, Any]:
        """
        Run multiple environment tests with different difficulty levels.
        
        Args:
            difficulties: List of difficulty levels to test
            mcp_config_path: Path to the MCP configuration file
            
        Returns:
            Dictionary with aggregated test results
        """
        logger.info(f"Running multiple environment tests with difficulties: {difficulties}")
        
        all_results = []
        for difficulty in difficulties:
            logger.info(f"Running test with difficulty {difficulty}")
            result = await self.run_single_test(difficulty, mcp_config_path)
            all_results.append(result)
            
            # Add delay between tests to avoid overwhelming the system
            await asyncio.sleep(1)
        
        # Generate summary
        summary = self._generate_test_summary(all_results)
        
        return {
            "test_results": all_results,
            "summary": summary
        }
    
    def _generate_test_summary(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of multiple test results."""
        successful_tests = [r for r in test_results if r["success"]]
        failed_tests = [r for r in test_results if not r["success"]]
        
        if not successful_tests:
            return {
                "total_tests": len(test_results),
                "successful_tests": 0,
                "failed_tests": len(failed_tests),
                "average_similarity": 0.0,
                "error": "No successful tests to analyze"
            }
        
        similarity_scores = [r["comparison_result"].get("similarity_score", 0.0) for r in successful_tests]
        
        summary = {
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "average_similarity": sum(similarity_scores) / len(similarity_scores),
            "min_similarity": min(similarity_scores),
            "max_similarity": max(similarity_scores),
            "scores_by_difficulty": {}
        }
        
        # Group scores by difficulty
        for result in successful_tests:
            difficulty = result["difficulty"]
            score = result["comparison_result"].get("similarity_score", 0.0)
            if difficulty not in summary["scores_by_difficulty"]:
                summary["scores_by_difficulty"][difficulty] = []
            summary["scores_by_difficulty"][difficulty].append(score)
        
        # Calculate averages by difficulty
        for difficulty, scores in summary["scores_by_difficulty"].items():
            summary["scores_by_difficulty"][difficulty] = {
                "average": sum(scores) / len(scores),
                "count": len(scores),
                "scores": scores
            }
        
        return summary 
    
    async def _initialize_simulated_environment(self, simulated_env: TestableSimulatedEnvironment, initial_state: Dict[str, str]):
        """Initialize the simulated environment with the same initial state as the real environment."""
        try:
            # Ensure tools are collected first
            if not simulated_env.tools:
                await simulated_env.collect_resources()
            
            # Create a text description of the initial state for the simulated environment
            state_description = "Initial filesystem state:\n"
            for file_path, content in initial_state.items():
                # Truncate content for readability
                content_preview = content[:200] + "..." if len(content) > 200 else content
                state_description += f"- {file_path}: {content_preview}\n"
            
            # Set the environment expectations to include the initial state
            simulated_env.environment_expectations = state_description
            
            # Initialize the simulated environment's state with the initial files
            simulated_env.state = [
                {
                    "tool_name": "filesystem_write_file",
                    "arguments": {"path": file_path, "content": content},
                    "response": f"File {file_path} created successfully"
                }
                for file_path, content in initial_state.items()
            ]
            
            logger.info(f"Initialized simulated environment with {len(initial_state)} files")
            
        except Exception as e:
            logger.error(f"Error initializing simulated environment: {e}")
            raise 