#!/usr/bin/env python3

import asyncio
import json
import os
from typing import Optional, Tuple, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax

from mcp.types import Prompt, Resource, Tool

from .config_parse import scan_mcp_config_file, check_server_with_timeout


def format_path_line(path: str, status: str) -> Text:
    """Format a path and status with colors"""
    text = Text()
    text.append(path, style="blue bold")
    text.append(" - ")
    text.append(status, style="green" if "found" in status else "red")
    return text


def format_schema(schema: dict[str, Any]) -> str:
    """Format a JSON schema into a readable string"""
    if not schema:
        return ""
    try:
        return json.dumps(schema, indent=2)
    except Exception:
        return str(schema)


async def scan_mcp_file(path: str, timeout: int = 10, suppress_io: bool = True, verbose: bool = True, output_file: Optional[str] = None) -> None:
    """Scan an MCP config file and print tools, resources and prompts

    Args:
        path: Path to the MCP config file
        timeout: Timeout in seconds for server connections
        suppress_io: Whether to suppress server IO
        verbose: Whether to show verbose output
        output_file: Optional path to save extracted entities as JSON
    """
    console = Console()

    try:
        servers = scan_mcp_config_file(path).get_servers()
        status = f"found {len(servers)} server{'' if len(servers) == 1 else 's'}"
    except FileNotFoundError:
        status = "file does not exist"
        console.print(format_path_line(path, status))
        return
    except Exception as e:
        status = f"could not parse file: {str(e)}"
        console.print(format_path_line(path, status))
        return

    if verbose:
        console.print(format_path_line(path, status))

    # Dictionary to store entities per server
    all_entities: dict[str, Tuple[list[Prompt], list[Resource], list[Tool]]] = {}

    # Scan all servers
    for server_name, server_config in servers.items():
        try:
            if verbose:
                console.print(f"Scanning server [cyan]{server_name}[/cyan]...")

            prompts, resources, tools = await check_server_with_timeout(
                server_config, timeout, suppress_io
            )

            all_entities[server_name] = (prompts, resources, tools)

            if verbose:
                console.print(f"  ✓ Found: [green]{len(prompts)}[/green] prompts, [green]{len(resources)}[/green] resources, [green]{len(tools)}[/green] tools")

        except asyncio.TimeoutError:
            console.print(f"  ✗ [red]Timed out[/red] connecting to server [cyan]{server_name}[/cyan]")
        except Exception as e:
            console.print(f"  ✗ [red]Error[/red] scanning server [cyan]{server_name}[/cyan]: {str(e)}")

    # Save to JSON file if output_file is specified
    if output_file:
        try:
            # Prepare JSON-serializable data
            json_data = {}
            for server_name, (prompts, resources, tools) in all_entities.items():
                json_data[server_name] = {
                    "prompts": [prompt.model_dump() for prompt in prompts],
                    "resources": [resource.model_dump() for resource in resources],
                    "tools": [tool.model_dump() for tool in tools]
                }

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

            # Write to file
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)

            if verbose:
                console.print(f"Entities saved to: [green]{output_file}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving to {output_file}: {str(e)}[/red]")

    # Print detailed information
    for server_name, (prompts, resources, tools) in all_entities.items():
        console.print(Panel(f"[bold cyan]{server_name}[/bold cyan]", expand=False))

        # Print prompts
        if prompts:
            prompt_table = Table(title="Prompts", show_lines=True)
            prompt_table.add_column("ID", style="cyan")
            prompt_table.add_column("Title", style="green")
            prompt_table.add_column("Description", style="yellow")

            for prompt in prompts:
                prompt_table.add_row(
                    prompt.id,
                    prompt.title or "",
                    prompt.description or ""
                )

            console.print(prompt_table)

        # Print resources
        if resources:
            resource_table = Table(title="Resources", show_lines=True)
            resource_table.add_column("ID", style="cyan")
            resource_table.add_column("Title", style="green")
            resource_table.add_column("Type", style="magenta")
            resource_table.add_column("Description", style="yellow")

            for resource in resources:
                resource_table.add_row(
                    resource.id,
                    resource.title or "",
                    resource.type,
                    resource.description or ""
                )

            console.print(resource_table)

        # Print tools
        if tools:
            tool_table = Table(title="Tools", show_lines=True)
            tool_table.add_column("Name", style="cyan")
            tool_table.add_column("Description", style="yellow")
            tool_table.add_column("Input Schema", style="green")
            tool_table.add_column("Properties", style="magenta")

            for tool in tools:
                # Extract additional properties that might be present
                additional_props = {}
                for key, value in tool.model_dump().items():
                    if key not in ["name", "description", "inputSchema"]:
                        additional_props[key] = value

                tool_table.add_row(
                    tool.name,
                    tool.description or "",
                    format_schema(getattr(tool, "inputSchema", {})),
                    format_schema(additional_props) if additional_props else ""
                )

            console.print(tool_table)

        console.print()