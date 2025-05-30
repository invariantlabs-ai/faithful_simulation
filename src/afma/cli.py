#!/usr/bin/env python3

import asyncio
import os
import sys
import json
import yaml
from typing import Optional, Tuple, List
from pathlib import Path

import click
from rich.console import Console
from loguru import logger
from tqdm.asyncio import tqdm

from .mcp_parser.scan_mcp import scan_mcp_file, scan_mcp_config_file, check_server_with_timeout
from .simulation.user import CombinatoricUserSet, User
from .simulation.agent import Agent
from .simulation.utils import get_tools_from_mcp_config
from .evaluation import Evaluator
from .evaluation.metrics import WeightedLevenshteinMetric
from .simulation.environment import SimulatedEnvironment, Environment


console = Console()


@click.group()
def cli():
    """AFMA Command Line Interface.

    Provides various utilities for working with MCP configs, AI models, and more.
    """
    pass


@cli.command(name="scan-mcp")
@click.argument("path", type=click.Path(exists=True))
@click.option("--timeout", "-t", type=int, default=10,
              help="Timeout in seconds for server connections")
@click.option("--show-server-output", "-s", is_flag=True,
              help="Show MCP server output")
@click.option("--output", "-o", type=click.Path(),
              help="Output JSON file to save extracted entities")
def scan_mcp_command(path: str, timeout: int, show_server_output: bool, output: Optional[str]):
    """Scan an MCP config file and display available entities.

    Connects to MCP servers defined in the config file and lists
    all available prompts, resources, and tools.
    """
    asyncio.run(scan_mcp_file(
        path=path,
        timeout=timeout,
        suppress_io=not show_server_output,
        output_file=output
    ))


@cli.command(name="create-users")
@click.argument("mcp_config_path", type=click.Path(exists=True))
@click.argument("litellm_config_path", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--max-permutations", "-m", type=int, default=3,
              help="Maximum length of tool permutations")
@click.option("--max-users-per-len", type=int, default=None,
              help="Maximum number of users to generate per permutation length")
@click.option("--timeout", "-t", type=int, default=10,
              help="Timeout in seconds for server connections")
def create_users_command(mcp_config_path: str, litellm_config_path: str, output: str, max_permutations: int,
                        timeout: int, max_users_per_len: Optional[int]):
    """Create users using CombinatoricUserSet with tools from MCP config.

    Connects to MCP servers defined in the config file, retrieves available tools,
    and generates users based on permutations of these tools.
    """
    async def run_create_users():
        # Load litellm config
        with open(litellm_config_path, 'r') as f:
            llm_config = json.load(f)

        try:
            # Get tools from MCP config file using the utility function
            all_tools = await get_tools_from_mcp_config(mcp_config_path, timeout)

            # Create user set
            user_set = CombinatoricUserSet(
                tools_info=all_tools,
                llm_config=llm_config,
                max_permutation_length=max_permutations,
                max_users_per_len=max_users_per_len
            )

            # Generate users
            logger.info("Generating users...")
            users = await user_set.generate_users()
            logger.success(f"Generated {len(users)} users")

            # Save users
            os.makedirs(Path(output).parent, exist_ok=True)
            users_results = [{
                "source": user.source,
                "user_goal": user.user_goal
            } for user in users]
            with open(output, 'w') as f:
                json.dump(users_results, f, indent=2)
            logger.success(f"Saved user histories to {output}")

        except FileNotFoundError:
            # Error already logged by get_tools_from_mcp_config if it's the mcp_config_path
            # This will catch if litellm_config_path is not found, or if mcp_config_path was not found by get_tools_from_mcp_config
            logger.error(f"Configuration file not found. Please check paths: MCP Config='{mcp_config_path}', LiteLLM Config='{litellm_config_path}'")
        except ValueError as e: # Catches "No tools found" from get_tools_from_mcp_config
            logger.error(f"Failed to create users: {str(e)}")
        except Exception as e:
            logger.exception("Error processing MCP config or creating users")

    # Run the async function
    asyncio.run(run_create_users())


@cli.command(name="simulate-chats")
@click.argument("mcp_config_path", type=click.Path(exists=True))
@click.argument("litellm_config_path", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--max-permutations", "-m", type=int, default=3,
              help="Maximum length of tool permutations")
@click.option("--max-users-per-len", type=int, default=None,
              help="Maximum number of users to generate per permutation length")
@click.option("--max-turns", "-t", type=int, default=10,
              help="Maximum number of conversation turns")
@click.option("--connect-timeout", type=int, default=10,
              help="Timeout in seconds for server connections during tool retrieval")
@click.option("--simulate-environment", "-s", is_flag=True,
              help="Use simulated environment instead of real tools")
@click.option("--concurrency", "-c", type=int, default=5,
              help="Maximum number of concurrent chat simulations")
def simulate_chats_command(mcp_config_path: str, litellm_config_path: str, output: str,
                           max_permutations: int, max_turns: int, connect_timeout: int, 
                           max_users_per_len: Optional[int], simulate_environment: bool,
                           concurrency: int):
    """Create users and simulate chats with an agent.

    Generates users based on permutations of tools and runs conversations with an agent.
    If --simulate-environment is used, tools will be simulated with LLM instead of real calls.
    """
    async def run_simulations():
        try:
            with open(litellm_config_path, 'r') as f:
                llm_config = json.load(f)
        except FileNotFoundError:
            logger.error(f"LiteLLM config file {litellm_config_path} not found. Cannot simulate chats.")
            return
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from LiteLLM config file {litellm_config_path}. Cannot simulate chats.")
            return

        all_tools = []
        try:
            all_tools = await get_tools_from_mcp_config(mcp_config_path, connect_timeout)
        except FileNotFoundError:
            logger.error(f"MCP config file {mcp_config_path} not found. Cannot simulate chats.")
            return
        except ValueError as e:
            logger.error(f"Failed to retrieve tools: {str(e)}. Cannot simulate chats.")
            return
        except Exception as e:
            logger.exception(f"Unexpected error retrieving tools from MCP config: {e}. Cannot simulate chats.")
            return

        # Create user set
        user_set = CombinatoricUserSet(
            tools_info=all_tools,
            llm_config=llm_config,
            max_permutation_length=max_permutations,
            max_users_per_len=max_users_per_len
        )

        logger.info("Generating users for simulation...")
        users = await user_set.generate_users()
        logger.success(f"Generated {len(users)} users for simulation")

        if not users:
            logger.warning("No users were generated. Skipping chat simulations.")
            return

        # Log whether we're using real or simulated environment
        if simulate_environment:
            logger.info("Using simulated environment for tool calls")
        else:
            logger.info("Using real environment for tool calls")

        # Create a semaphore to limit concurrent simulations
        semaphore = asyncio.Semaphore(concurrency)
        logger.info(f"Running simulations with concurrency limit of {concurrency}")

        async def run_single_simulation(user_index, user):
            async with semaphore:
                # Create environment based on simulation mode
                if simulate_environment:
                    environment = SimulatedEnvironment(mcp_config_path, llm_config, connect_timeout, user_goal=user.user_goal)
                else:
                    environment = Environment(mcp_config_path, connect_timeout)
                
                agent = Agent(llm_config, environment)
                logger.info(f"Running chat simulation {user_index+1}/{len(users)} with user goal: '{user.user_goal}'")
                history = await user.talk_with(agent, max_turns=max_turns)
                return {
                    "user_goal": user.user_goal, 
                    "user_source": user.source, 
                    "used_tools": agent.get_used_tools(), 
                    "history": history
                }

        # Create tasks for all simulations
        tasks = [run_single_simulation(i, user) for i, user in enumerate(users)]
        
        # Run all tasks concurrently and collect results with progress bar
        chat_histories = await tqdm.gather(
            *tasks,
            desc="Simulating conversations",
            unit="conv",
            total=len(tasks)
        )

        if output:
            with open(output, 'w') as f:
                json.dump(chat_histories, f, indent=2)

        return chat_histories

    asyncio.run(run_simulations())


@cli.command(name="evaluate")
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("conversations_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(),
              help="Output path for detailed evaluation results")
def evaluate_command(config_path: str, conversations_path: str, output: Optional[str]):
    """Evaluate conversations using the configuration file.
    
    Loads conversations from a JSON file and evaluates them using metrics
    defined in the configuration. Supports personality-based analysis.
    """
    async def run_evaluation():
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load conversations
            with open(conversations_path, 'r') as f:
                conversations = json.load(f)
            
            logger.info(f"Loaded {len(conversations)} conversations for evaluation")
            
            # Extract evaluation config
            eval_config = config.get("evaluation", {})
            llm_config = eval_config.get("litellm", {})
            concurrency = eval_config.get("concurrency", 5)
            
            # Check if weighted Levenshtein is requested
            use_weighted_levenshtein = eval_config["use_weighted_levenshtein"]
            tool_definitions = None
            embedding_config = eval_config["embedding"]
            
            if use_weighted_levenshtein:
                # Need to load tools from MCP config
                mcp_config_path = config["agent"]["toolset"]
                if mcp_config_path and os.path.exists(mcp_config_path):
                    timeout = config["simulation"]["timeout"]
                    try:
                        all_tools = await get_tools_from_mcp_config(mcp_config_path, timeout)
                        tool_definitions = {tool["name"]: tool for tool in all_tools}
                        logger.info(f"Loaded {len(tool_definitions)} tool definitions for weighted Levenshtein")
                    except Exception as e:
                        logger.warning(f"Failed to load tools for weighted Levenshtein: {e}")
                        use_weighted_levenshtein = False
                else:
                    logger.warning("No MCP config found for weighted Levenshtein, disabling")
                    use_weighted_levenshtein = False
            
            # Create evaluator with optional weighted Levenshtein support
            evaluator = Evaluator(
                llm_config=llm_config,
                tool_definitions=tool_definitions,
                embedding_config=embedding_config if use_weighted_levenshtein else None,
                use_weighted_levenshtein=use_weighted_levenshtein
            )
            
            logger.info("Starting evaluation...")
            summary = await evaluator.evaluate_batch(
                conversations, 
                concurrency=concurrency
            )
            
            # Print summary
            evaluator.print_summary(summary)
            
            # Save detailed results if requested
            if output:
                evaluator.save_detailed_results(summary, output)
                logger.success(f"Detailed results saved to {output}")
            elif eval_config.get("output_detailed_results", False):
                # Generate default output path
                default_output = conversations_path.replace('.json', '_evaluation_results.json')
                evaluator.save_detailed_results(summary, default_output)
                logger.success(f"Detailed results saved to {default_output}")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON conversations: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during evaluation: {e}")
    
    asyncio.run(run_evaluation())


@cli.command(name="run-pipeline")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), default="./results",
              help="Output directory for all results")
@click.option("--skip-simulation", is_flag=True,
              help="Skip simulation and only run evaluation on existing conversations")
@click.option("--conversations-file", type=click.Path(),
              help="Path to existing conversations file (used with --skip-simulation)")
def run_pipeline_command(config_path: str, output_dir: str, skip_simulation: bool, conversations_file: Optional[str]):
    """Run the complete evaluation pipeline from configuration.
    
    This command orchestrates the entire process:
    1. Generate users with different personalities
    2. Run simulations with different environment personalities  
    3. Evaluate results with comprehensive metrics
    4. Generate detailed reports
    """
    async def run_complete_pipeline():
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            conversations_path = conversations_file
            
            if not skip_simulation:
                # Extract configuration sections
                user_generation_config = config.get("user_generation", {})
                environment_config = config.get("environment", {})
                agent_config = config.get("agent", {})
                simulation_config = config.get("simulation", {})
                
                # Get MCP config path
                mcp_config_path = agent_config.get("toolset")
                if not mcp_config_path or not os.path.exists(mcp_config_path):
                    logger.error(f"MCP config file not found: {mcp_config_path}")
                    return
                
                # Get tools
                timeout = simulation_config["timeout"]
                all_tools = await get_tools_from_mcp_config(mcp_config_path, timeout)
                
                # Generate users with personalities
                user_personalities = user_generation_config.get("personalities", [])
                user_set = CombinatoricUserSet(
                    tools_info=all_tools,
                    llm_config=user_generation_config["litellm"],
                    max_permutation_length=user_generation_config["max_permutations"],
                    max_users_per_len=user_generation_config["max_users_per_len"],
                    semaphore_limit=user_generation_config["semaphore_limit"],
                    personalities=user_personalities
                )
                
                logger.info("Generating users...")
                users = await user_set.generate_users()
                logger.success(f"Generated {len(users)} users")
                
                # Save user profiles
                user_profiles_path = os.path.join(output_dir, "user_profiles.json")
                user_profiles = [{
                    "user_goal": user.user_goal,
                    "source": user.source,
                    "personality": getattr(user, 'personality', None),
                    "max_turns": user.max_turns
                } for user in users]
                with open(user_profiles_path, 'w') as f:
                    json.dump(user_profiles, f, indent=2)
                logger.success(f"Saved {len(user_profiles)} user profiles to {user_profiles_path}")
                
                # Run simulations with different environment personalities
                environment_personalities = environment_config["simulated_qualities"]
                use_simulated_env = environment_config["simulated"]
                
                all_conversations = []
                concurrency = simulation_config["concurrency"]
                max_turns = simulation_config["max_turns"]
                
                # Create semaphore for simulation concurrency
                semaphore = asyncio.Semaphore(concurrency)
                
                async def run_single_simulation(user, env_personality_info):
                    async with semaphore:
                        env_personality = env_personality_info.get("description") if env_personality_info else None
                        env_name = env_personality_info.get("name") if env_personality_info else "default"
                        
                        # Create environment based on simulation mode
                        if use_simulated_env:
                            environment = SimulatedEnvironment(mcp_config_path, agent_config.get("litellm", {}), timeout, env_personality, user.user_goal)
                        else:
                            from .simulation.environment import Environment
                            environment = Environment(mcp_config_path, timeout)
                        
                        agent = Agent(agent_config.get("litellm", {}), environment)
                        
                        try:
                            history = await user.talk_with(agent, max_turns=max_turns)
                        except Exception as e:
                            logger.error(f"Error in simulation: {e}")
                            return None
                        
                        return {
                            "user_goal": user.user_goal,
                            "user_source": user.source,
                            "used_tools": agent.get_used_tools(),
                            "history": history,
                            "user_personality": getattr(user, 'personality', None),
                            "environment_personality": env_personality
                        }
                
                # Create all simulation tasks
                simulation_tasks = []
                for user in users:
                    if use_simulated_env and environment_personalities:
                        # Run with each environment personality
                        for env_personality_info in environment_personalities:
                            simulation_tasks.append(run_single_simulation(user, env_personality_info))
                    else:
                        # Run with default environment
                        simulation_tasks.append(run_single_simulation(user, None))
                
                logger.info(f"Running {len(simulation_tasks)} simulations with concurrency {concurrency}")
                
                # Use tqdm to show progress for conversation generation
                all_conversations = await tqdm.gather(
                    *simulation_tasks,
                    desc="Generating conversations",
                    unit="conv",
                    total=len(simulation_tasks)
                )
                logger.info(f"All conversations: {len(all_conversations)}")
                all_conversations = [conv for conv in all_conversations if conv is not None]
                logger.info(f"All conversations after filtering: {len(all_conversations)}")
                
                # Save raw conversations
                conversations_path = os.path.join(output_dir, "conversations.json")
                with open(conversations_path, 'w') as f:
                    json.dump(all_conversations, f, indent=2)
                logger.success(f"Saved {len(all_conversations)} conversations to {conversations_path}")
                
                # Save conversation summaries for easier analysis
                conversation_summaries_path = os.path.join(output_dir, "conversation_summaries.json")
                conversation_summaries = [{
                    "conversation_id": i,
                    "user_goal": conv["user_goal"],
                    "user_personality": conv.get("user_personality"),
                    "environment_personality": conv.get("environment_personality"),
                    "expected_tools": [tool["name"] for tool in conv["user_source"]],
                    "used_tools": conv["used_tools"],
                    "tool_sequence_length": len(conv["user_source"]),
                    "conversation_length": len(conv["history"]),
                    "tools_match": conv["used_tools"] == [tool["name"] for tool in conv["user_source"]]
                } for i, conv in enumerate(all_conversations)]
                with open(conversation_summaries_path, 'w') as f:
                    json.dump(conversation_summaries, f, indent=2)
                logger.success(f"Saved conversation summaries to {conversation_summaries_path}")
            
            else:
                # Load existing conversations
                if not conversations_file or not os.path.exists(conversations_file):
                    logger.error("Must provide --conversations-file when using --skip-simulation")
                    return
                
                with open(conversations_file, 'r') as f:
                    all_conversations = json.load(f)
                logger.info(f"Loaded {len(all_conversations)} existing conversations")
            
            # Run evaluation
            eval_config = config.get("evaluation", {})
            
            # Check if weighted Levenshtein is requested
            use_weighted_levenshtein = eval_config["use_weighted_levenshtein"]
            tool_definitions = None
            embedding_config = eval_config["embedding"]
            
            if use_weighted_levenshtein:
                # Get tool definitions for semantic similarity
                if not skip_simulation:
                    # Use tools from simulation
                    tool_definitions = {tool["name"]: tool for tool in all_tools}
                else:
                    # Need to load tools from MCP config
                    mcp_config_path = config["agent"]["toolset"]
                    if mcp_config_path and os.path.exists(mcp_config_path):
                        timeout = config["simulation"]["timeout"]
                        try:
                            all_tools = await get_tools_from_mcp_config(mcp_config_path, timeout)
                            tool_definitions = {tool["name"]: tool for tool in all_tools}
                            logger.info(f"Loaded {len(tool_definitions)} tool definitions for weighted Levenshtein")
                        except Exception as e:
                            logger.warning(f"Failed to load tools for weighted Levenshtein: {e}")
                            use_weighted_levenshtein = False
                    else:
                        logger.warning("No MCP config found for weighted Levenshtein, disabling")
                        use_weighted_levenshtein = False
            
            # Create evaluator with optional weighted Levenshtein support
            evaluator = Evaluator(
                llm_config=eval_config["litellm"],
                tool_definitions=tool_definitions,
                embedding_config=embedding_config,
                use_weighted_levenshtein=use_weighted_levenshtein
            )
            
            logger.info("Starting evaluation...")
            summary = await evaluator.evaluate_batch(
                all_conversations, 
                concurrency=eval_config.get("concurrency", 5)
            )
            
            # Add all metric scores to conversations for viewer
            logger.info("Adding metric scores to conversations...")
            for metric_name, metric_details in summary.metric_details.items():
                for detail in metric_details:
                    conv_id = detail["conversation_id"]
                    score = detail["score"]
                    all_conversations[conv_id][f"{metric_name}_score"] = score
                    
                    # Also add detailed results if available
                    if detail.get("details"):
                        all_conversations[conv_id][f"{metric_name}_details"] = detail["details"]
            
            # Save updated conversations with all metric scores
            conversations_path = os.path.join(output_dir, "conversations.json")
            with open(conversations_path, 'w') as f:
                json.dump(all_conversations, f, indent=2)
            logger.success(f"Updated conversations with all metric scores")
            
            # Print and save results
            evaluator.print_summary(summary)
            
            eval_results_path = os.path.join(output_dir, "evaluation_results.json")
            evaluator.save_detailed_results(summary, eval_results_path)
            
            logger.success(f"Pipeline completed! Results saved in {output_dir}")
            
        except Exception as e:
            logger.exception(f"Error in pipeline execution: {e}")
    
    asyncio.run(run_complete_pipeline())


@cli.command(name="compare-tool-sequences")
@click.argument("toolset_path", type=click.Path(exists=True))
@click.argument("sequence1", type=str)
@click.argument("sequence2", type=str)
@click.option("--embedding-model", "-e", type=str, default="text-embedding-3-small",
              help="Embedding model to use for semantic similarity")
@click.option("--timeout", "-t", type=int, default=10,
              help="Timeout in seconds for server connections")
@click.option("--verbose", "-v", is_flag=True,
              help="Show detailed calculation steps")
def compare_tool_sequences_command(toolset_path: str, sequence1: str, sequence2: str, 
                                 embedding_model: str, 
                                 timeout: int, verbose: bool):
    """Compare two tool sequences using weighted Levenshtein metric.
    
    Takes two tool sequences (comma-separated tool names) and calculates
    the weighted Levenshtein distance between them using semantic similarity
    of tool descriptions.
    
    Example:
        afma compare-tool-sequences config.json "tool1,tool2,tool3" "tool1,tool3,tool2"
    """
    async def run_comparison():
        try:
            # Load tool definitions from MCP config
            logger.info(f"Loading tools from {toolset_path}...")
            all_tools = await get_tools_from_mcp_config(toolset_path, timeout)
            tool_definitions = {tool["name"]: tool for tool in all_tools}
            logger.success(f"Loaded {len(tool_definitions)} tool definitions")
            
            # Parse tool sequences
            tools1 = [tool.strip() for tool in sequence1.split(",") if tool.strip()]
            tools2 = [tool.strip() for tool in sequence2.split(",") if tool.strip()]
            
            if verbose:
                logger.info(f"Sequence 1: {tools1}")
                logger.info(f"Sequence 2: {tools2}")
            
            # Validate that all tools exist in the toolset
            all_sequence_tools = set(tools1 + tools2)
            missing_tools = all_sequence_tools - set(tool_definitions.keys())
            if missing_tools:
                logger.error(f"Unknown tools in sequences: {missing_tools}")
                logger.info(f"Available tools: {list(tool_definitions.keys())}")
                return
            
            # Create embedding config
            embedding_config = {
                "model": embedding_model
            }
            
            # Create WeightedLevenshteinMetric
            logger.info("Initializing weighted Levenshtein metric...")
            metric = WeightedLevenshteinMetric(
                tool_definitions=tool_definitions,
                embedding_config=embedding_config
            )
            
            # Calculate the metric
            logger.info("Calculating weighted Levenshtein distance...")
            result = await metric._weighted_levenshtein_distance(tools1, tools2)
            
            # Convert distance to similarity score (0-1 range)
            max_len = max(len(tools1), len(tools2)) if tools1 or tools2 else 1
            similarity_score = 1.0 - (result / max_len) if max_len > 0 else 1.0
            
            # Print results
            console.print("\n[bold]Tool Sequence Comparison Results[/bold]")
            console.print(f"Sequence 1: {', '.join(tools1)}")
            console.print(f"Sequence 2: {', '.join(tools2)}")
            console.print(f"[bold red]Raw Distance: {result:.4f}[/bold red]")
            console.print(f"[bold green]Similarity Score: {similarity_score:.4f}[/bold green]")
            
            if verbose:
                console.print("\n[bold]Additional Details:[/bold]")
                console.print(f"Sequence 1 length: {len(tools1)}")
                console.print(f"Sequence 2 length: {len(tools2)}")
                console.print(f"Maximum length: {max_len}")
                console.print(f"Distance normalized by max length: {result/max_len:.4f}")
            
            if verbose and hasattr(result, 'details'):
                console.print("\n[bold]Detailed Calculation:[/bold]")
                console.print(result.details)
            
        except FileNotFoundError:
            logger.error(f"Toolset file not found: {toolset_path}")
        except ValueError as e:
            logger.error(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error during comparison: {e}")
    
    asyncio.run(run_comparison())


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()