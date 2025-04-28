#!/usr/bin/env python3

import asyncio
import os
import sys
from typing import Optional

import click
from rich.console import Console

from .mcp_parser.scan_mcp import scan_mcp_file


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
@click.option("--quiet", "-q", is_flag=True, 
              help="Reduce verbosity")
def scan_mcp_command(path: str, timeout: int, show_server_output: bool, quiet: bool):
    """Scan an MCP config file and display available entities.
    
    Connects to MCP servers defined in the config file and lists
    all available prompts, resources, and tools.
    """
    asyncio.run(scan_mcp_file(
        path=path,
        timeout=timeout,
        suppress_io=not show_server_output,
        verbose=not quiet
    ))

def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main() 