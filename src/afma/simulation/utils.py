import asyncio
from typing import List, Dict, Any

from loguru import logger

from ..mcp_parser.scan_mcp import scan_mcp_config_file, check_server_with_timeout


async def get_tools_from_mcp_config(mcp_config_path: str, timeout: int) -> List[Dict[str, Any]]:
    """
    Scans an MCP config file, connects to servers, and retrieves all available tools.

    Args:
        mcp_config_path: Path to the MCP configuration file.
        timeout: Timeout in seconds for server connections.
        quiet: If True, suppress informational logging.

    Returns:
        A list of tool dictionaries.

    Raises:
        FileNotFoundError: If the mcp_config_path does not exist.
        ValueError: If no tools are found after checking all servers.
        Exception: For other unexpected errors during tool retrieval.
    """
    all_tools: List[Dict[str, Any]] = []
    try:
        server_configs = scan_mcp_config_file(mcp_config_path).get_servers()
        logger.info(f"Found {len(server_configs)} server(s) in MCP config at '{mcp_config_path}'")

        for server_name, server_config_values in server_configs.items():
            try:
                logger.info(f"Retrieving tools from server '{server_name}'...")

                _prompts, _resources, tools_from_server = await check_server_with_timeout(
                    server_config_values, timeout, suppress_mcpserver_io=True
                )

                for tool in tools_from_server:
                    tool_dict = tool.model_dump()
                    tool_dict["server"] = server_name
                    tool_dict["original_name"] = tool_dict["name"]
                    tool_dict["name"] = f"{server_name}_{tool_dict['name']}"
                    all_tools.append(tool_dict)

                if tools_from_server:
                    logger.success(f"Found {len(tools_from_server)} tools from server '{server_name}'")
                else:
                    logger.info(f"No tools found from server '{server_name}'")

            except asyncio.TimeoutError:
                logger.error(f"Timed out connecting to server '{server_name}' after {timeout}s")
            except Exception as e:
                logger.exception(f"Error scanning server '{server_name}': {e}")
                # Continue to try other servers

        if not all_tools:
            msg = "No tools found in MCP config after checking all servers."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Total tools collected: {len(all_tools)}")

        return all_tools

    except FileNotFoundError:
        logger.error(f"MCP config file not found: {mcp_config_path}")
        raise  # Re-raise for the caller to handle
    except Exception as e:
        logger.exception(f"Unexpected error while retrieving tools from MCP config '{mcp_config_path}': {e}")
        raise # Re-raise for the caller to handle