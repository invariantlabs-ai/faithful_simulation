"""
Concurrent test runner for environment simulation testing with session-based isolation.
"""

import json
import os
import asyncio
import uuid
import statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger
import yaml
from datetime import datetime

from .environment_wrapper import TestableEnvironment, TestableSimulatedEnvironment
from .state_generator import EnvironmentStateGenerator
from .task_generator import TaskGenerator
from .state_comparator import StateComparator


class ConcurrentEnvironmentTestRunner:
    """Concurrent test runner for environment simulation testing with session isolation."""
    
    def __init__(self, config_path: str = "afma_testing/environment/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.test_config = self.config["environment_testing"]
        self.llm_config = self.test_config["litellm"]
        self.grid_config = self.test_config["grid_testing"]
        
        # Initialize components
        self.state_generator = EnvironmentStateGenerator(self.llm_config, self.test_config)
        self.task_generator = TaskGenerator(self.llm_config, self.test_config)
        self.state_comparator = StateComparator(self.test_config)
        
        # Concurrency control
        self.max_concurrent_tests = self.test_config.get("test", {}).get("max_concurrent_tests", 4)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tests)
        
        # Results storage
        self.results_dir = Path(self.test_config.get("test", {}).get("results_output_dir", "results/environment_testing"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def _modify_mcp_config_for_session(self, session_id: str, base_config_path: str = "agent_mcp_configs/files.json") -> str:
        """
        Create a modified MCP config for the session with the correct file_playground path.
        
        Args:
            session_id: Unique session identifier
            base_config_path: Path to the base MCP configuration file
            
        Returns:
            Path to the modified MCP configuration file
        """
        try:
            # Load the base config
            with open(base_config_path, 'r') as f:
                config = json.load(f)
            
            # Modify the file_playground path to include session ID
            session_path = f"file_playground/session_{session_id}/"
            config["mcpServers"]["filesystem"]["args"][2] = session_path
            
            # Create a temporary config file for this session
            temp_config_path = f"agent_mcp_configs/files_session_{session_id}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.debug(f"Created session MCP config: {temp_config_path} with path: {session_path}")
            return temp_config_path
            
        except Exception as e:
            logger.error(f"Failed to modify MCP config for session {session_id}: {e}")
            raise
    
    def _create_session_directory(self, session_id: str):
        """Create the session directory before initializing MCP servers."""
        try:
            session_dir = Path(f"file_playground/session_{session_id}")
            session_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created session directory: {session_dir}")
        except Exception as e:
            logger.error(f"Failed to create session directory for {session_id}: {e}")
            raise
    
    def _cleanup_session_config(self, session_config_path: str):
        """Clean up the temporary session MCP config file."""
        try:
            if os.path.exists(session_config_path):
                os.remove(session_config_path)
                logger.debug(f"Cleaned up session config: {session_config_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session config {session_config_path}: {e}")
    
    async def _cleanup_session_directory(self, session_id: str):
        """Clean up the session directory."""
        try:
            session_dir = Path(f"file_playground/session_{session_id}")
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
                logger.debug(f"Cleaned up session directory: {session_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session directory for {session_id}: {e}")
    
    async def run_single_test(self, file_count: int, task_complexity: int, run_number: int) -> Dict[str, Any]:
        """
        Run a single environment test with the specified parameters.
        
        Args:
            file_count: Number of files to create in the initial state
            task_complexity: Number of tool calls in the task sequence
            run_number: Run number for this configuration
            
        Returns:
            Dictionary with test results
        """
        session_id = str(uuid.uuid4())[:8]  # Short session ID
        session_config_path = None
        
        async with self.semaphore:  # Limit concurrent tests
            logger.info(f"Starting test: files={file_count}, complexity={task_complexity}, run={run_number}, session={session_id}")
            
            test_result = {
                "session_id": session_id,
                "file_count": file_count,
                "task_complexity": task_complexity,
                "run_number": run_number,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": None,
                "initial_state": {},
                "task_description": "",
                "tool_sequence": [],
                "real_final_state": {},
                "simulated_final_state": {},
                "comparison_result": {},
                "execution_log": [],
                "execution_time": 0.0
            }
            
            start_time = datetime.now()
            
            try:
                # Create session-specific MCP config
                session_config_path = self._modify_mcp_config_for_session(session_id)
                
                # Create session directory before initializing environments
                self._create_session_directory(session_id)
                
                # Initialize environments with session config
                session_dir = f"file_playground/session_{session_id}"
                real_env = TestableEnvironment(session_config_path, timeout=10, llm_config=self.llm_config, session_dir=session_dir)
                simulated_env = TestableSimulatedEnvironment(session_config_path, self.llm_config, timeout=10, session_dir=session_dir)
                
                # Step 1: Generate and create initial environment state
                logger.info(f"[{session_id}] Step 1: Generating initial environment state with {file_count} files")
                initial_state = await self.state_generator.generate_environment_state(file_count, real_env, session_dir=session_dir)
                if not initial_state:
                    raise Exception("Failed to generate initial environment state")
                
                test_result["initial_state"] = initial_state
                test_result["execution_log"].append("Initial environment state created successfully")
                
                # Step 2: Get initial state from real environment
                logger.info(f"[{session_id}] Step 2: Getting initial state from real environment")
                real_initial_state = await real_env.get_state()
                test_result["execution_log"].append(f"Retrieved initial state with {len(real_initial_state)} files")
                
                # Step 2.5: Initialize simulated environment with the same initial state
                logger.info(f"[{session_id}] Step 2.5: Initializing simulated environment with initial state")
                await self._initialize_simulated_environment(simulated_env, real_initial_state, session_id)
                test_result["execution_log"].append(f"Initialized simulated environment with {len(real_initial_state)} files")
                
                # Step 3: Generate task and tool sequence
                logger.info(f"[{session_id}] Step 3: Generating task and tool sequence with complexity {task_complexity}")
                
                # Get available tools from the real environment
                await real_env.collect_resources()
                available_tools = real_env.tools if real_env.tools else []
                
                task_description, tool_sequence = await self.task_generator.generate_task_and_sequence(
                    real_initial_state, 
                    available_tools,
                    max_complexity=task_complexity,
                    session_dir=session_dir
                )
                
                test_result["task_description"] = task_description
                test_result["tool_sequence"] = tool_sequence
                
                logger.info(f"[{session_id}] Generated task: {task_description}")
                logger.info(f"[{session_id}] Generated {len(tool_sequence)} tool calls")
                
                # Step 4: Execute tool sequence in both environments
                logger.info(f"[{session_id}] Step 4: Executing tool sequence in both environments")
                
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
                        logger.debug(f"[{session_id}] Real environment tool call {i+1}/{len(tool_sequence)} completed")
                    except Exception as e:
                        logger.error(f"[{session_id}] Real environment tool call {i+1} failed: {e}")
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
                        logger.debug(f"[{session_id}] Simulated environment tool call {i+1}/{len(tool_sequence)} completed")
                    except Exception as e:
                        logger.error(f"[{session_id}] Simulated environment tool call {i+1} failed: {e}")
                        simulated_execution_results.append({
                            "tool_call": tool_call,
                            "result": str(e),
                            "success": False
                        })
                
                test_result["real_execution_results"] = real_execution_results
                test_result["simulated_execution_results"] = simulated_execution_results
                test_result["execution_log"].append("Tool sequence executed in both environments")
                
                # Step 5: Get final states from both environments
                logger.info(f"[{session_id}] Step 5: Getting final states from both environments")
                real_final_state = await real_env.get_state()
                simulated_final_state = await simulated_env.get_state()
                
                test_result["real_final_state"] = real_final_state
                test_result["simulated_final_state"] = simulated_final_state
                test_result["execution_log"].append(f"Retrieved final states: real={len(real_final_state)} files, simulated={len(simulated_final_state)} files")
                
                # Step 6: Compare states
                logger.info(f"[{session_id}] Step 6: Comparing environment states")
                comparison_result = await self.state_comparator.compare_states(
                    real_final_state, simulated_final_state, task_description
                )
                
                test_result["comparison_result"] = comparison_result
                test_result["execution_log"].append(f"Comparison completed with similarity score: {comparison_result.get('similarity_score', 0.0):.3f}")
                
                # Step 7: Cleanup (if enabled)
                if self.test_config.get("test", {}).get("cleanup_after_test", True):
                    logger.info(f"[{session_id}] Step 7: Cleaning up test environment")
                    await self.state_generator.cleanup_environment(real_env)
                    await self._cleanup_session_directory(session_id)
                    test_result["execution_log"].append("Environment cleanup completed")
                
                test_result["success"] = True
                logger.success(f"[{session_id}] Test completed successfully with similarity score: {comparison_result.get('similarity_score', 0.0):.3f}")
                
            except Exception as e:
                logger.exception(f"[{session_id}] Test failed: {e}")
                test_result["error"] = str(e)
                test_result["execution_log"].append(f"Test failed with error: {e}")
            
            finally:
                # Cleanup session config
                if session_config_path:
                    self._cleanup_session_config(session_config_path)
                
                # Calculate execution time
                end_time = datetime.now()
                test_result["execution_time"] = (end_time - start_time).total_seconds()
            
            return test_result
    
    async def run_grid_tests(self) -> Dict[str, Any]:
        """
        Run grid tests with all combinations of file counts and task complexities.
        
        Returns:
            Dictionary with all test results and summary
        """
        file_counts = self.grid_config.get("file_counts", [3, 5, 8])
        task_complexities = self.grid_config.get("task_complexities", [3, 5, 8])
        runs_per_config = self.grid_config.get("runs_per_config", 3)
        
        logger.info(f"Starting grid tests: {len(file_counts)} file counts × {len(task_complexities)} complexities × {runs_per_config} runs = {len(file_counts) * len(task_complexities) * runs_per_config} total tests")
        
        # Create all test configurations
        test_configs = []
        for file_count in file_counts:
            for task_complexity in task_complexities:
                for run_number in range(1, runs_per_config + 1):
                    test_configs.append((file_count, task_complexity, run_number))
        
        # Run all tests concurrently
        tasks = [self.run_single_test(file_count, task_complexity, run_number) 
                for file_count, task_complexity, run_number in test_configs]
        
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"Test {i} failed with exception: {result}")
                processed_results.append({
                    "session_id": f"failed_{i}",
                    "file_count": test_configs[i][0],
                    "task_complexity": test_configs[i][1],
                    "run_number": test_configs[i][2],
                    "success": False,
                    "error": str(result),
                    "execution_time": 0.0
                })
            else:
                processed_results.append(result)
        
        # Generate summary and save results
        summary = self._generate_grid_test_summary(processed_results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"grid_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "config": self.config,
                "test_configs": test_configs,
                "results": processed_results,
                "summary": summary
            }, f, indent=2, default=str)
        
        logger.success(f"Grid tests completed. Results saved to {results_file}")
        
        return {
            "test_results": processed_results,
            "summary": summary,
            "results_file": str(results_file)
        }
    
    def _generate_grid_test_summary(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive summary of grid test results."""
        successful_tests = [r for r in test_results if r.get("success", False)]
        failed_tests = [r for r in test_results if not r.get("success", False)]
        
        summary = {
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(test_results) if test_results else 0.0,
            "average_execution_time": sum(r.get("execution_time", 0.0) for r in test_results) / len(test_results) if test_results else 0.0,
            "results_by_configuration": {},
            "similarity_scores": {}
        }
        
        if successful_tests:
            similarity_scores = [r.get("comparison_result", {}).get("similarity_score", 0.0) for r in successful_tests]
            summary["similarity_scores"] = {
                "average": sum(similarity_scores) / len(similarity_scores),
                "min": min(similarity_scores),
                "max": max(similarity_scores),
                "all_scores": similarity_scores
            }
        
        # Group results by configuration
        for result in test_results:
            config_key = f"{result.get('file_count', 0)}_files_{result.get('task_complexity', 0)}_complexity"
            if config_key not in summary["results_by_configuration"]:
                summary["results_by_configuration"][config_key] = {
                    "file_count": result.get("file_count", 0),
                    "task_complexity": result.get("task_complexity", 0),
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "similarity_scores": [],
                    "average_execution_time": 0.0
                }
            
            config_summary = summary["results_by_configuration"][config_key]
            config_summary["total_runs"] += 1
            
            if result.get("success", False):
                config_summary["successful_runs"] += 1
                similarity_score = result.get("comparison_result", {}).get("similarity_score", 0.0)
                config_summary["similarity_scores"].append(similarity_score)
            else:
                config_summary["failed_runs"] += 1
            
            config_summary["average_execution_time"] += result.get("execution_time", 0.0)
        
        # Calculate averages for each configuration
        for config_summary in summary["results_by_configuration"].values():
            if config_summary["total_runs"] > 0:
                config_summary["average_execution_time"] /= config_summary["total_runs"]
                if config_summary["similarity_scores"]:
                    config_summary["average_similarity"] = sum(config_summary["similarity_scores"]) / len(config_summary["similarity_scores"])
                    # Calculate standard deviation
                    if len(config_summary["similarity_scores"]) > 1:
                        config_summary["std_similarity"] = statistics.stdev(config_summary["similarity_scores"])
                    else:
                        config_summary["std_similarity"] = 0.0
                else:
                    config_summary["average_similarity"] = 0.0
                    config_summary["std_similarity"] = 0.0
        
        return summary
    
    async def _initialize_simulated_environment(self, simulated_env: TestableSimulatedEnvironment, initial_state: Dict[str, str], session_id: str):
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
                    "arguments": {"path": f"file_playground/session_{session_id}/{file_path}", "content": content},
                    "response": f"File {file_path} created successfully"
                }
                for file_path, content in initial_state.items()
            ]
            
            logger.info(f"Initialized simulated environment with {len(initial_state)} files")
            
        except Exception as e:
            logger.error(f"Error initializing simulated environment: {e}")
            raise 