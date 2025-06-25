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
from .evaluation.metrics import WeightedLevenshteinMetric, GoalAchievementMetric
from .simulation.environment import SimulatedEnvironment
from litellm.caching.caching import Cache
import litellm

litellm.cache = Cache(type="disk")

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

            # Save users and tool graph
            os.makedirs(Path(output).parent, exist_ok=True)
            users_results = [{
                "source": user.source,
                "user_goal": user.user_goal,
                "environment_expectations": user.environment_expectations
            } for user in users]
            
            # Create output with both users and tool graph
            output_data = {
                "users": users_results,
                "tool_graph": user_set._tool_graph,
                "metadata": {
                    "total_users": len(users),
                    "permutation_lengths": parsed_lengths,
                    "max_users_per_len": max_users_per_len,
                    "tools_count": len(all_tools),
                    "tool_names": [tool["name"] for tool in all_tools]
                }
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.success(f"Saved {len(users)} user histories and tool graph to {output}")

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
                        environment_expectations=profile["environment_expectations"],
                        llm_config=user_config["litellm"],
                        source=profile["source"],
                        max_turns=profile["max_turns"],
                        personality=profile.get("personality"),
                        personality_name=profile.get("user_personality_name")
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
                    "environment_expectations": user.environment_expectations,
                    "source": user.source,
                    "personality": getattr(user, 'personality', None),
                    "user_personality_name": user.personality_name,
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
                        "user_personality_name": conv.get("user_personality_name"),
                        "environment_personality": conv.get("environment_personality"),
                        "environment_personality_name": conv.get("environment_personality_name"),
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
                            environment = SimulatedEnvironment(mcp_config_path, agent_config.get("litellm", {}), timeout, env_personality, user.environment_expectations)
                        else:
                            from .simulation.environment import Environment
                            environment = Environment(mcp_config_path, timeout)
                        
                        agent = Agent(agent_config.get("litellm", {}), environment)
                        
                        try:
                            history = await user.talk_with(agent, max_turns=max_turns)
                        except Exception as e:
                            logger.error(f"Error in simulation: {e}")
                            return None
                        
                        # Capture environment state for goal achievement metric
                        environment_state = None
                        if use_simulated_env and hasattr(environment, 'state'):
                            environment_state = environment.state
                        
                        return {
                            "user_goal": user.user_goal,
                            "user_source": user.source,
                            "used_tools": agent.get_used_tools(),
                            "history": history,
                            "user_personality": getattr(user, 'personality', None),
                            "user_personality_name": user.personality_name,
                            "environment_personality": env_personality,
                            "environment_personality_name": env_name,
                            "environment_state": environment_state,
                            "environment_expectations": user.environment_expectations,
                            "trace_set_id": trace_set_id,
                            "instantiation_id": instantiation_id,
                            "trace_set_metadata": {
                                "user_goal": user.user_goal,
                                "user_personality": getattr(user, 'personality', None),
                                "user_personality_name": user.personality_name,
                                "environment_personality": env_personality,
                                "environment_personality_name": env_name,
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
                                    environment_expectations=user.environment_expectations,
                                    llm_config=user.llm_config,
                                    source=user.source,
                                    max_turns=user.max_turns,
                                    personality=getattr(user, 'personality', None),
                                    personality_name=user.personality_name
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
                                environment_expectations=user.environment_expectations,
                                llm_config=user.llm_config,
                                source=user.source,
                                max_turns=user.max_turns,
                                personality=user.personality,
                                personality_name=user.personality_name
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
                    "user_personality_name": conv.get("user_personality_name"),
                    "environment_personality": conv.get("environment_personality"),
                    "environment_personality_name": conv.get("environment_personality_name"),
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
            evaluation_config = config.get("env_goal_achievement", {"enabled": True})
            alignments_path = os.path.join(output_dir, "trace_alignments.json")
            alignment_summary_path = os.path.join(output_dir, "alignment_summary.json")
            
            if os.path.exists(alignments_path) and os.path.exists(alignment_summary_path):
                logger.info(f"Loading existing trace alignments from {alignments_path}")
                logger.success("Trace alignments already computed")
            else:
                await compute_trace_alignments(all_conversations, all_tools, trace_alignment_config, output_dir, evaluation_config)
            
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


async def compute_trace_alignments(conversations: List[Dict[str, Any]], all_tools: List[Dict[str, Any]], 
                                 alignment_config: Dict[str, Any], output_dir: str, 
                                 evaluation_config: Optional[Dict[str, Any]] = None):
    """Compute trace alignments for conversation sets with multiple instantiations."""
    logger.info("Computing trace alignments...")
    
    # Create a mapping from (trace_set_id, instantiation_id) to conversation_id for efficient lookup
    conversation_id_map = {}
    for i, conv in enumerate(conversations):
        trace_set_id = conv.get("trace_set_id")
        instantiation_id = conv.get("instantiation_id")
        if trace_set_id and instantiation_id is not None:
            conversation_id_map[(trace_set_id, instantiation_id)] = i
    
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
    
    # Create GoalAchievementMetric for goal achievement evaluation
    # Use evaluation config if provided, otherwise use defaults
    if evaluation_config and evaluation_config.get("enabled", True):
        goal_achievement_llm_config = evaluation_config["litellm"]
        goal_metric = GoalAchievementMetric(llm_config=goal_achievement_llm_config)
        enable_goal_achievement = True
        # Create semaphore for goal achievement evaluation concurrency
        goal_eval_concurrency = evaluation_config["concurrency"]
        goal_eval_semaphore = asyncio.Semaphore(goal_eval_concurrency)
    else:
        goal_metric = None
        enable_goal_achievement = False
        goal_eval_semaphore = None
    
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
                conversation_id_map.get((trace_set_id, conv.get("instantiation_id")), -1)
                for conv in convs
            ]
            alignment_result["instantiation_count"] = len(convs)
            
            # Compute goal achievement metrics for each conversation if enabled
            goal_achievement_results = []
            if enable_goal_achievement and goal_metric:
                logger.info(f"Computing goal achievement metrics for {len(convs)} conversations in trace set {trace_set_id}")
                async def evaluate_goal_achievement(conv):
                    async with goal_eval_semaphore:
                        try:
                            goal_result = await goal_metric.evaluate(
                                user_goal=conv["user_goal"],
                                environment_state=conv.get("environment_state", []),
                                environment_expectations=conv.get("environment_expectations", "No specific environment expectations provided")
                            )
                            return {
                                "conversation_id": conversation_id_map.get((trace_set_id, conv.get("instantiation_id")), -1),
                                "score": goal_result.score,
                                "details": goal_result.details,
                                "error": goal_result.error
                            }
                        except Exception as e:
                            logger.error(f"Error computing goal achievement for conversation in trace set {trace_set_id}: {e}")
                            return {
                                "conversation_id": conversation_id_map.get((trace_set_id, conv.get("instantiation_id")), -1),
                                "score": 0.0,
                                "details": {},
                                "error": str(e)
                            }
                
                # Run goal achievement evaluations concurrently
                goal_eval_tasks = [evaluate_goal_achievement(conv) for conv in convs]
                goal_achievement_results = await asyncio.gather(*goal_eval_tasks)
                logger.success(f"Completed goal achievement evaluation for trace set {trace_set_id}")
            
            # Add goal achievement results to alignment result
            alignment_result["goal_achievement_results"] = goal_achievement_results
            
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
        goal_results = alignment_data.get("goal_achievement_results", [])
        
        # Calculate alignment statistics using similarity scores from metrics
        similarities = [a["similarity"] for a in alignments]
        distances = [a["distance"] for a in alignments]
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        min_similarity = min(similarities) if similarities else 0
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        max_distance = max(distances) if distances else 0
        min_distance = min(distances) if distances else 0
        
        # Calculate goal achievement statistics
        goal_scores = [r["score"] for r in goal_results if r["error"] is None]
        avg_goal_achievement = sum(goal_scores) / len(goal_scores) if goal_scores else 0
        max_goal_achievement = max(goal_scores) if goal_scores else 0
        min_goal_achievement = min(goal_scores) if goal_scores else 0
        
        alignment_summary.append({
            "trace_set_id": trace_set_id,
            "user_goal": metadata["user_goal"],
            "user_personality": metadata["user_personality"],
            "user_personality_name": metadata.get("user_personality_name"),
            "environment_personality": metadata["environment_personality"],
            "environment_personality_name": metadata.get("environment_personality_name"),
            "expected_tools": metadata["expected_tools"],
            "instantiation_count": alignment_data["instantiation_count"],
            "reference_sequence": alignment_data["reference_sequence"],
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "avg_distance": avg_distance,
            "max_distance": max_distance,
            "min_distance": min_distance,
            "avg_goal_achievement": avg_goal_achievement,
            "max_goal_achievement": max_goal_achievement,
            "min_goal_achievement": min_goal_achievement,
            "conversation_ids": alignment_data["conversation_ids"]
        })
    
    # Save alignment summary
    summary_path = os.path.join(output_dir, "alignment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(alignment_summary, f, indent=2)
    logger.success(f"Saved alignment summary to {summary_path}")
    
    # Log summary statistics
    if enable_goal_achievement:
        all_goal_scores = [ts["avg_goal_achievement"] for ts in alignment_summary if ts["avg_goal_achievement"] > 0]
        if all_goal_scores:
            overall_avg_goal = sum(all_goal_scores) / len(all_goal_scores)
            logger.success(f"Goal achievement evaluation completed. Overall average: {overall_avg_goal:.3f}")
        else:
            logger.warning("No valid goal achievement scores found")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()