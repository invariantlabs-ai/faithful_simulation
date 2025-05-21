#!/usr/bin/env python3

import asyncio
import os
import sys
import json
from typing import Optional, Tuple, List
from pathlib import Path

import click
from rich.console import Console
from loguru import logger

from .mcp_parser.scan_mcp import scan_mcp_file, scan_mcp_config_file, check_server_with_timeout
from .simulation.user import CombinatoricUserSet, User
from .simulation.agent import Agent
from .simulation.utils import get_tools_from_mcp_config


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
def simulate_chats_command(mcp_config_path: str, litellm_config_path: str, output: str,
                           max_permutations: int, max_turns: int, connect_timeout: int, max_users_per_len: Optional[int]):
    """Create users and simulate chats with an agent.

    Generates users based on permutations of tools and runs conversations with an agent.
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

        chat_histories = []
        for i, user in enumerate(users):
            agent = Agent(mcp_config_path=mcp_config_path, llm_config=llm_config)
            logger.info(f"Running chat simulation {i+1}/{len(users)} with user goal: '{user.user_goal}'")
            history = await user.talk_with(agent, max_turns=max_turns)
            chat_histories.append({"user_goal": user.user_goal, "history": history})

        if output:
            with open(output, 'w') as f:
                json.dump(chat_histories, f, indent=2)

        return chat_histories

    asyncio.run(run_simulations())


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()