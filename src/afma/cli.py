#!/usr/bin/env python3

import asyncio
import os
import sys
import json
import yaml
from typing import Optional, Tuple, List, Dict, Any
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
@click.option("--permutation-lengths", "-p", type=str, default="1,2,3",
              help="Comma-separated list of permutation lengths (e.g., '1,2,3,4')")
@click.option("--max-users-per-len", type=int, default=None,
              help="Maximum number of users to generate per permutation length")
@click.option("--timeout", "-t", type=int, default=10,
              help="Timeout in seconds for server connections")
def create_users_command(mcp_config_path: str, litellm_config_path: str, output: str, permutation_lengths: str,
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

            # Parse permutation lengths
            parsed_lengths = [int(x.strip()) for x in permutation_lengths.split(',')]

            # Create user set (using same config for both generation and simulation for backward compatibility)
            user_set = CombinatoricUserSet(
                tools_info=all_tools,
                generation_llm_config=llm_config,
                simulation_llm_config=llm_config,
                permutation_lengths=parsed_lengths,
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


@cli.command(name="run-pipeline")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), default="./results",
              help="Output directory for all results")
def run_pipeline_command(config_path: str, output_dir: str):
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
            
            
            # Extract configuration sections
            user_generation_config = config["user_generation"]
            environment_config = config["environment"]
            agent_config = config["agent"]
            simulation_config = config["simulation"]
            
            # Get MCP config path
            mcp_config_path = agent_config["toolset"]
            if not mcp_config_path or not os.path.exists(mcp_config_path):
                logger.error(f"MCP config file not found: {mcp_config_path}")
                return
            
            # Get tools
            timeout = simulation_config["timeout"]
            all_tools = await get_tools_from_mcp_config(mcp_config_path, timeout)
            
            # Check if user profiles already exist
            user_profiles_path = os.path.join(output_dir, "user_profiles.json")
            users = []
            
            if os.path.exists(user_profiles_path):
                logger.info(f"Loading existing user profiles from {user_profiles_path}")
                with open(user_profiles_path, 'r') as f:
                    user_profiles = json.load(f)
                
                # Reconstruct User objects from saved profiles
                user_config = config["user"]
                for profile in user_profiles:
                    user = User(
                        user_goal=profile["user_goal"],
                        llm_config=user_config["litellm"],
                        source=profile["source"],
                        max_turns=profile["max_turns"],
                        personality=profile.get("personality")
                    )
                    users.append(user)
                logger.success(f"Loaded {len(users)} users from existing profiles")
            else:
                # Generate users with personalities
                user_config = config["user"]
                user_personalities = user_generation_config["personalities"]
                user_set = CombinatoricUserSet(
                    tools_info=all_tools,
                    generation_llm_config=user_generation_config["litellm"],
                    simulation_llm_config=user_config["litellm"],
                    permutation_lengths=user_generation_config["permutation_lengths"],
                    max_users_per_len=user_generation_config["max_users_per_len"],
                    semaphore_limit=user_generation_config["semaphore_limit"],
                    personalities=user_personalities
                )
                
                logger.info("Generating users...")
                users = await user_set.generate_users()
                logger.success(f"Generated {len(users)} users")
                
                # Save user profiles
                user_profiles = [{
                    "user_goal": user.user_goal,
                    "source": user.source,
                    "personality": getattr(user, 'personality', None),
                    "max_turns": user.max_turns
                } for user in users]
                with open(user_profiles_path, 'w') as f:
                    json.dump(user_profiles, f, indent=2)
                logger.success(f"Saved {len(user_profiles)} user profiles to {user_profiles_path}")
            
            # Check if conversations already exist
            conversations_path = os.path.join(output_dir, "conversations.json")
            conversation_summaries_path = os.path.join(output_dir, "conversation_summaries.json")
            
            if os.path.exists(conversations_path):
                logger.info(f"Loading existing conversations from {conversations_path}")
                with open(conversations_path, 'r') as f:
                    all_conversations = json.load(f)
                logger.success(f"Loaded {len(all_conversations)} conversations from existing file")
                
                # Also load conversation summaries if they exist, otherwise regenerate them
                if os.path.exists(conversation_summaries_path):
                    logger.info(f"Loading existing conversation summaries from {conversation_summaries_path}")
                else:
                    logger.info("Regenerating conversation summaries from loaded conversations")
                    conversation_summaries = [{
                        "conversation_id": i,
                        "user_goal": conv["user_goal"],
                        "user_personality": conv.get("user_personality"),
                        "environment_personality": conv.get("environment_personality"),
                        "expected_tools": [tool["name"] for tool in conv["user_source"]],
                        "used_tools": conv["used_tools"],
                        "tool_sequence_length": len(conv["user_source"]),
                        "conversation_length": len(conv["history"]),
                        "tools_match": conv["used_tools"] == [tool["name"] for tool in conv["user_source"]],
                        "trace_set_id": conv.get("trace_set_id"),
                        "instantiation_id": conv.get("instantiation_id")
                    } for i, conv in enumerate(all_conversations)]
                    with open(conversation_summaries_path, 'w') as f:
                        json.dump(conversation_summaries, f, indent=2)
                    logger.success(f"Saved conversation summaries to {conversation_summaries_path}")
            else:
                # Run simulations with different environment personalities
                environment_personalities = environment_config["simulated_qualities"]
                use_simulated_env = environment_config["simulated"]
                instantiations_per_trace = simulation_config["instantiations_per_trace"]
                
                all_conversations = []
                concurrency = simulation_config["concurrency"]
                max_turns = simulation_config["max_turns"]
                
                # Create semaphore for simulation concurrency
                semaphore = asyncio.Semaphore(concurrency)
                
                async def run_single_simulation(user, env_personality_info, trace_set_id, instantiation_id):
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
                            "environment_personality": env_personality,
                            "trace_set_id": trace_set_id,
                            "instantiation_id": instantiation_id,
                            "trace_set_metadata": {
                                "user_goal": user.user_goal,
                                "user_personality": getattr(user, 'personality', None),
                                "environment_personality": env_personality,
                                "expected_tools": [tool["name"] for tool in user.source]
                            }
                        }
                    
                # Create all simulation tasks with trace set tracking
                simulation_tasks = []
                trace_set_counter = 0
                
                for user in users:
                    if use_simulated_env and environment_personalities:
                        # Run with each environment personality
                        for env_personality_info in environment_personalities:
                            trace_set_id = f"trace_set_{trace_set_counter}"
                            
                            # Run multiple instantiations for the same trace set
                            for instantiation_id in range(instantiations_per_trace):
                                # Create a fresh copy of the user for each instantiation
                                fresh_user = User(
                                    user_goal=user.user_goal,
                                    llm_config=user.llm_config,
                                    source=user.source,
                                    max_turns=user.max_turns,
                                    personality=getattr(user, 'personality', None)
                                )
                                simulation_tasks.append(run_single_simulation(fresh_user, env_personality_info, trace_set_id, instantiation_id))
                            
                            trace_set_counter += 1
                    else:
                        # Run with default environment
                        trace_set_id = f"trace_set_{trace_set_counter}"
                        
                        # Run multiple instantiations for the same trace set
                        for instantiation_id in range(instantiations_per_trace):
                            # Create a fresh copy of the user for each instantiation
                            fresh_user = User(
                                user_goal=user.user_goal,
                                llm_config=user.llm_config,
                                source=user.source,
                                max_turns=user.max_turns,
                                personality=getattr(user, 'personality', None)
                            )
                            simulation_tasks.append(run_single_simulation(fresh_user, None, trace_set_id, instantiation_id))
                        
                        trace_set_counter += 1
                
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
                with open(conversations_path, 'w') as f:
                    json.dump(all_conversations, f, indent=2)
                logger.success(f"Saved {len(all_conversations)} conversations to {conversations_path}")
                
                # Save conversation summaries for easier analysis
                conversation_summaries = [{
                    "conversation_id": i,
                    "user_goal": conv["user_goal"],
                    "user_personality": conv.get("user_personality"),
                    "environment_personality": conv.get("environment_personality"),
                    "expected_tools": [tool["name"] for tool in conv["user_source"]],
                    "used_tools": conv["used_tools"],
                    "tool_sequence_length": len(conv["user_source"]),
                    "conversation_length": len(conv["history"]),
                    "tools_match": conv["used_tools"] == [tool["name"] for tool in conv["user_source"]],
                    "trace_set_id": conv.get("trace_set_id"),
                    "instantiation_id": conv.get("instantiation_id")
                } for i, conv in enumerate(all_conversations)]
                with open(conversation_summaries_path, 'w') as f:
                    json.dump(conversation_summaries, f, indent=2)
                logger.success(f"Saved conversation summaries to {conversation_summaries_path}")
            
            # Compute trace alignments if they don't already exist
            trace_alignment_config = config.get("trace_alignment", {"min_instantiations": 2, "store_alignment_details": True})
            alignments_path = os.path.join(output_dir, "trace_alignments.json")
            alignment_summary_path = os.path.join(output_dir, "alignment_summary.json")
            
            if os.path.exists(alignments_path) and os.path.exists(alignment_summary_path):
                logger.info(f"Loading existing trace alignments from {alignments_path}")
                logger.success("Trace alignments already computed")
            else:
                await compute_trace_alignments(all_conversations, all_tools, trace_alignment_config, output_dir)
            
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
            
            # Calculate the metric with full alignment information
            logger.info("Calculating weighted Levenshtein distance and optimal alignment...")
            alignment_result = await metric.get_optimal_alignment(tools1, tools2)
            
            distance = alignment_result["distance"]
            similarity = alignment_result["similarity"]
            alignment = alignment_result["alignment"]
            operations = alignment_result["operations"]
            
            # Print results
            console.print("\n[bold]Tool Sequence Comparison Results[/bold]")
            console.print(f"Sequence 1: {', '.join(tools1)}")
            console.print(f"Sequence 2: {', '.join(tools2)}")
            console.print(f"[bold red]Raw Distance: {distance:.4f}[/bold red]")
            console.print(f"[bold green]Similarity Score: {similarity:.4f}[/bold green]")
            
            # Print optimal alignment
            console.print("\n[bold]Optimal Alignment:[/bold]")
            for i, (tool1, tool2) in enumerate(alignment):
                if tool1 is None:
                    console.print(f"  {i+1:2d}: [red]--- → {tool2}[/red] (insert)")
                elif tool2 is None:
                    console.print(f"  {i+1:2d}: [red]{tool1} → ---[/red] (delete)")
                elif tool1 == tool2:
                    console.print(f"  {i+1:2d}: [green]{tool1} → {tool2}[/green] (match)")
                else:
                    console.print(f"  {i+1:2d}: [yellow]{tool1} → {tool2}[/yellow] (substitute)")
            
            if verbose:
                console.print("\n[bold]Additional Details:[/bold]")
                console.print(f"Sequence 1 length: {len(tools1)}")
                console.print(f"Sequence 2 length: {len(tools2)}")
                max_len = max(len(tools1), len(tools2)) if tools1 or tools2 else 1
                console.print(f"Maximum length: {max_len}")
                console.print(f"Distance normalized by max length: {distance/max_len:.4f}")
                
                console.print("\n[bold]Operation Details:[/bold]")
                operation_counts = {"match": 0, "substitute": 0, "insert": 0, "delete": 0}
                for op, tool1, tool2 in operations:
                    operation_counts[op] += 1
                    if op == "match":
                        console.print(f"  [green]Match:[/green] {tool1}")
                    elif op == "substitute":
                        console.print(f"  [yellow]Substitute:[/yellow] {tool1} → {tool2}")
                    elif op == "insert":
                        console.print(f"  [red]Insert:[/red] {tool2}")
                    elif op == "delete":
                        console.print(f"  [red]Delete:[/red] {tool1}")
                
                console.print("\n[bold]Operation Summary:[/bold]")
                for op_type, count in operation_counts.items():
                    if count > 0:
                        console.print(f"  {op_type.capitalize()}: {count}")
                console.print(f"  Total operations: {sum(operation_counts.values())}")
            
        except FileNotFoundError:
            logger.error(f"Toolset file not found: {toolset_path}")
        except ValueError as e:
            logger.error(f"Error: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error during comparison: {e}")
    
    asyncio.run(run_comparison())


@cli.command(name="debug-tool-sequence")
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("tool_sequence", type=str)
@click.option("--output-dir", "-o", type=click.Path(), default="./debug_results",
              help="Output directory for debug results")
def debug_tool_sequence_command(config_path: str, tool_sequence: str, output_dir: str):
    """Debug a specific tool sequence with focused simulation.
    
    Takes a specific tool sequence (comma-separated tool names) and runs
    targeted simulations for debugging purposes. Much faster than full pipeline
    since it focuses on just one tool combination.
    
    Iterations, user personality, and environment personality are read from the config file.
    Trace alignments are automatically computed for analysis.
    
    Example:
        afma debug-tool-sequence config.yaml "tool1,tool2,tool3"
    """
    async def run_debug_sequence():
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract configuration sections
            user_generation_config = config["user_generation"]
            environment_config = config["environment"]
            agent_config = config["agent"]
            simulation_config = config["simulation"]
            
            # Get MCP config path
            mcp_config_path = agent_config["toolset"]
            if not mcp_config_path or not os.path.exists(mcp_config_path):
                logger.error(f"MCP config file not found: {mcp_config_path}")
                return
            
            # Get tools from MCP config
            timeout = simulation_config["timeout"]
            all_tools = await get_tools_from_mcp_config(mcp_config_path, timeout)
            tool_definitions = {tool["name"]: tool for tool in all_tools}
            
            # Parse and validate tool sequence
            requested_tools = [tool.strip() for tool in tool_sequence.split(",") if tool.strip()]
            
            # Validate that all requested tools exist
            missing_tools = set(requested_tools) - set(tool_definitions.keys())
            if missing_tools:
                logger.error(f"Unknown tools in sequence: {missing_tools}")
                logger.info(f"Available tools: {list(tool_definitions.keys())}")
                return
            
            # Get tool info for the specific sequence
            tool_info = [tool_definitions[tool_name] for tool_name in requested_tools]
            
            logger.info(f"Debug sequence: {requested_tools}")
            logger.info(f"Tool info loaded: {len(tool_info)} tools")
            
            # Read iterations from config (default to 1 if not specified)
            iterations = simulation_config.get("debug_iterations", 1)
            logger.info(f"Running {iterations} debug iterations")
            
            # Get personalities from config (same as run-pipeline)
            user_config = config["user"]
            user_personalities = user_generation_config.get("personalities", [None])
            environment_personalities = environment_config.get("simulated_qualities", [None])
            instantiations_per_trace = simulation_config.get("instantiations_per_trace", iterations)
            
            # Create users with all personalities but using our specific tool sequence
            debug_users = []
            for user_personality_info in user_personalities:
                user_personality = user_personality_info.get("description") if user_personality_info else None
                
                # Create a single user with the specified tool sequence
                user_set = CombinatoricUserSet(
                    tools_info=all_tools,
                    generation_llm_config=user_generation_config["litellm"],
                    simulation_llm_config=user_config["litellm"],
                    permutation_lengths=[len(requested_tools)],  # Only the length of our sequence
                    max_users_per_len=1,  # Only generate one user
                    personalities=[user_personality_info] if user_personality_info else None
                )
                
                # Generate and customize user
                users = await user_set.generate_users()
                if users:
                    debug_user = users[0]
                    debug_user.source = tool_info
                    debug_user.user_goal = f"Use the following tools in sequence: {', '.join(requested_tools)}"
                    debug_users.append(debug_user)
            
            if not debug_users:
                logger.error("Failed to generate any debug users")
                return
            
            logger.success(f"Created {len(debug_users)} debug users with tool sequence: {requested_tools}")
            
            # Save debug user profiles
            user_profiles_path = os.path.join(output_dir, "user_profiles.json")
            user_profiles = [{
                "user_goal": user.user_goal,
                "tool_sequence": requested_tools,
                "source": user.source,
                "personality": getattr(user, 'personality', None),
                "max_turns": user.max_turns
            } for user in debug_users]
            with open(user_profiles_path, 'w') as f:
                json.dump(user_profiles, f, indent=2)
            logger.success(f"Saved debug user profiles to {user_profiles_path}")
            
            # Run simulations following run-pipeline structure
            use_simulated_env = environment_config["simulated"]
            max_turns = simulation_config["max_turns"]
            concurrency = simulation_config["concurrency"]
            
            # Create semaphore for simulation concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def run_single_simulation(user, env_personality_info, trace_set_id, instantiation_id):
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
                        "environment_personality": env_personality,
                        "trace_set_id": trace_set_id,
                        "instantiation_id": instantiation_id,
                        "trace_set_metadata": {
                            "user_goal": user.user_goal,
                            "user_personality": getattr(user, 'personality', None),
                            "environment_personality": env_personality,
                            "expected_tools": [tool["name"] for tool in user.source]
                        },
                        "debug_metadata": {
                            "expected_tools": requested_tools,
                            "actual_tools": agent.get_used_tools(),
                            "sequence_length_match": len(agent.get_used_tools()) == len(requested_tools),
                            "exact_sequence_match": agent.get_used_tools() == requested_tools,
                            "tools_match": agent.get_used_tools() == requested_tools
                        }
                    }
            
            # Create all simulation tasks with trace set tracking (same as run-pipeline)
            simulation_tasks = []
            trace_set_counter = 0
            
            for user in debug_users:
                if use_simulated_env and environment_personalities:
                    # Run with each environment personality
                    for env_personality_info in environment_personalities:
                        trace_set_id = f"trace_set_{trace_set_counter}"
                        
                        # Run multiple instantiations for the same trace set
                        for instantiation_id in range(instantiations_per_trace):
                            # Create a fresh copy of the user for each instantiation
                            fresh_user = User(
                                user_goal=user.user_goal,
                                llm_config=user.llm_config,
                                source=user.source,
                                max_turns=user.max_turns,
                                personality=getattr(user, 'personality', None)
                            )
                            simulation_tasks.append(run_single_simulation(fresh_user, env_personality_info, trace_set_id, instantiation_id))
                        
                        trace_set_counter += 1
                else:
                    # Run with default environment
                    trace_set_id = f"trace_set_{trace_set_counter}"
                    
                    # Run multiple instantiations for the same trace set
                    for instantiation_id in range(instantiations_per_trace):
                        # Create a fresh copy of the user for each instantiation
                        fresh_user = User(
                            user_goal=user.user_goal,
                            llm_config=user.llm_config,
                            source=user.source,
                            max_turns=user.max_turns,
                            personality=getattr(user, 'personality', None)
                        )
                        simulation_tasks.append(run_single_simulation(fresh_user, None, trace_set_id, instantiation_id))
                    
                    trace_set_counter += 1
            
            logger.info(f"Running {len(simulation_tasks)} simulations with concurrency {concurrency}")
            
            # Use tqdm to show progress for conversation generation
            all_conversations = await tqdm.gather(
                *simulation_tasks,
                desc="Generating conversations",
                unit="conv",
                total=len(simulation_tasks)
            )
            all_conversations = [conv for conv in all_conversations if conv is not None]
            
            # Save conversations
            conversations_path = os.path.join(output_dir, "conversations.json")
            with open(conversations_path, 'w') as f:
                json.dump(all_conversations, f, indent=2)
            logger.success(f"Saved {len(all_conversations)} conversations to {conversations_path}")
            
            # Save conversation summaries for easier analysis (same as run-pipeline)
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
                "tools_match": conv["debug_metadata"]["exact_sequence_match"],
                "trace_set_id": conv.get("trace_set_id"),
                "instantiation_id": conv.get("instantiation_id"),
                "target_sequence": requested_tools
            } for i, conv in enumerate(all_conversations)]
            with open(conversation_summaries_path, 'w') as f:
                json.dump(conversation_summaries, f, indent=2)
            logger.success(f"Saved conversation summaries to {conversation_summaries_path}")
            
            # Compute trace alignments if enabled (same as run-pipeline)
            trace_alignment_config = config.get("trace_alignment", {})
            if trace_alignment_config.get("enabled", False):
                await compute_trace_alignments(all_conversations, all_tools, trace_alignment_config, output_dir)
            
            # Print quick summary
            console.print(f"\n[bold]Results for tool sequence: {', '.join(requested_tools)}[/bold]")
            console.print(f"Conversations completed: {len(all_conversations)}")
            exact_matches = sum(1 for conv in all_conversations if conv["debug_metadata"]["exact_sequence_match"])
            console.print(f"Exact sequence matches: {exact_matches}")
            success_rate = exact_matches / len(all_conversations) if all_conversations else 0
            console.print(f"Success rate: {success_rate:.2%}")
            
            # Show tool usage patterns
            tool_usage_analysis = {}
            for conv in all_conversations:
                actual_sequence = conv["used_tools"]
                sequence_key = ",".join(actual_sequence)
                if sequence_key not in tool_usage_analysis:
                    tool_usage_analysis[sequence_key] = 0
                tool_usage_analysis[sequence_key] += 1
            
            if tool_usage_analysis:
                console.print("\n[bold]Tool usage patterns:[/bold]")
                for sequence, count in tool_usage_analysis.items():
                    console.print(f"  {sequence}: {count} times")
            
            # Compute trace alignments (always enabled now)
            if all_conversations:
                trace_alignment_config = config.get("trace_alignment", {"min_instantiations": 2, "store_alignment_details": True})
                await compute_trace_alignments(all_conversations, all_tools, trace_alignment_config, output_dir)
            
            logger.success(f"Tool sequence testing completed! Results saved in {output_dir}")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
        except Exception as e:
            logger.exception(f"Error in debug sequence execution: {e}")
    
    asyncio.run(run_debug_sequence())


async def compute_trace_alignments(conversations: List[Dict[str, Any]], all_tools: List[Dict[str, Any]], 
                                 alignment_config: Dict[str, Any], output_dir: str):
    """Compute trace alignments for conversation sets with multiple instantiations."""
    logger.info("Computing trace alignments...")
    
    # Group conversations by trace set ID
    trace_sets = {}
    for conv in conversations:
        trace_set_id = conv.get("trace_set_id")
        if trace_set_id:
            if trace_set_id not in trace_sets:
                trace_sets[trace_set_id] = []
            trace_sets[trace_set_id].append(conv)
    
    logger.info(f"Found {len(trace_sets)} trace sets")
    
    # Filter trace sets that have enough instantiations
    min_instantiations = alignment_config["min_instantiations"]
    valid_trace_sets = {
        trace_set_id: convs for trace_set_id, convs in trace_sets.items() 
        if len(convs) >= min_instantiations
    }
    
    logger.info(f"Found {len(valid_trace_sets)} trace sets with at least {min_instantiations} instantiations")
    
    if not valid_trace_sets:
        logger.warning("No trace sets with sufficient instantiations for alignment")
        return
    
    # Create embedding config for weighted Levenshtein
    embedding_config = {"model": "text-embedding-3-small"}
    tool_definitions = {tool["name"]: tool for tool in all_tools}
    
    # Create WeightedLevenshteinMetric for alignment computation
    metric = WeightedLevenshteinMetric(
        tool_definitions=tool_definitions,
        embedding_config=embedding_config
    )
    
    # Compute alignments for each trace set
    all_alignments = {}
    
    for trace_set_id, convs in valid_trace_sets.items():
        logger.info(f"Computing alignment for trace set {trace_set_id} ({len(convs)} instantiations)")
        
        # Extract expected tool sequence from trace set metadata
        trace_set_metadata = convs[0]["trace_set_metadata"]
        expected_tools = trace_set_metadata["expected_tools"]
        
        # Extract actual tool sequences
        actual_tool_sequences = [conv["used_tools"] for conv in convs]
        
        # Compute alignment using expected tools as reference
        try:
            alignment_result = await metric.align_multiple_sequences(expected_tools, actual_tool_sequences)
            
            # Add metadata
            alignment_result["trace_set_metadata"] = trace_set_metadata
            alignment_result["conversation_ids"] = [
                next(i for i, c in enumerate(conversations) if c.get("trace_set_id") == trace_set_id and c.get("instantiation_id") == conv.get("instantiation_id"))
                for conv in convs
            ]
            alignment_result["instantiation_count"] = len(convs)
            
            all_alignments[trace_set_id] = alignment_result
            
        except Exception as e:
            logger.error(f"Error computing alignment for trace set {trace_set_id}: {e}")
            continue
    
    # Save alignments
    alignments_path = os.path.join(output_dir, "trace_alignments.json")
    with open(alignments_path, 'w') as f:
        json.dump(all_alignments, f, indent=2)
    logger.success(f"Saved trace alignments to {alignments_path}")
    
    # Create alignment summary
    alignment_summary = []
    for trace_set_id, alignment_data in all_alignments.items():
        metadata = alignment_data["trace_set_metadata"]
        alignments = alignment_data["alignments"]
        
        # Calculate alignment statistics using similarity scores from metrics
        similarities = [a["similarity"] for a in alignments]
        distances = [a["distance"] for a in alignments]
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        min_similarity = min(similarities) if similarities else 0
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        max_distance = max(distances) if distances else 0
        min_distance = min(distances) if distances else 0
        
        alignment_summary.append({
            "trace_set_id": trace_set_id,
            "user_goal": metadata["user_goal"],
            "user_personality": metadata["user_personality"],
            "environment_personality": metadata["environment_personality"],
            "expected_tools": metadata["expected_tools"],
            "instantiation_count": alignment_data["instantiation_count"],
            "reference_sequence": alignment_data["reference_sequence"],
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "avg_distance": avg_distance,
            "max_distance": max_distance,
            "min_distance": min_distance,
            "conversation_ids": alignment_data["conversation_ids"]
        })
    
    # Save alignment summary
    summary_path = os.path.join(output_dir, "alignment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(alignment_summary, f, indent=2)
    logger.success(f"Saved alignment summary to {summary_path}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()