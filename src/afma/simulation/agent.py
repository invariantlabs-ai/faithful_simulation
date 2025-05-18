from typing import Optional, Any
import litellm
from loguru import logger
import json

from mcp import ClientSession
from mcp.types import Prompt, Resource, Tool, TextContent, ImageContent, EmbeddedResource

from afma.mcp_parser import scan_mcp_config_file, get_client


SYSTEM_PROMPT_AGENT = (
    "Help the user with their question using the tools available to you. If you can't help, say so. If you think you did everything that you can, write CONVERSATION_COMPLETE. Do it both after successful completion of a user's request and after you are sure you can do nothing for a user because user is acting weird or you don't have access to what user is asking for."
)

class Agent:
    def __init__(self, llm_config: dict[str, Any], mcp_config_path: str, timeout: int = 10):
        self.llm_config = llm_config
        self.message_history: list[dict[str, str]] = []
        self.server_configs = scan_mcp_config_file(mcp_config_path).get_servers()
        self.timeout = timeout
        self.tools = None
        self.prompts = None
        self.resources = None
        self.server_by_tool_name = {}

    def _convert_mcp_tools_to_openai_format(self, mcp_tools: list[Any] | dict[str, Any]) -> list[dict[str, Any]]:
        """Convert MCP tool format to OpenAI tool format"""
        openai_tools = []

        if hasattr(mcp_tools, 'tools'):
            tools_list = mcp_tools.tools
            logger.debug("Found ListToolsResult, extracting tools attribute")
        elif isinstance(mcp_tools, dict):
            tools_list = mcp_tools.get('tools', [])
        else:
            tools_list = mcp_tools

        for tool in tools_list:
            logger.debug(f"Processing tool: {tool}, type: {type(tool)}")
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                openai_name = tool.name # self._sanitize_tool_name(tool.name)
                # self.tool_name_mapping[openai_name] = tool.name
                logger.debug(f"Tool has required attributes. Name: {tool.name}")

                tool_schema = getattr(tool, 'inputSchema', {
                    "type": "object",
                    "properties": {},
                    "required": []
                })

                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": openai_name,
                        "description": tool.description,
                        "parameters": tool_schema
                    }
                }
                openai_tools.append(openai_tool)
            else:
                logger.debug(f"Tool missing required attributes: has name = {hasattr(tool, 'name')}, has description = {hasattr(tool, 'description')}")

        return openai_tools

    def _sanitize_tool_name(self, name: str) -> str:
        return name.replace("-", "_").replace(" ", "_").lower()

    async def _collect_resources(self):
        """Collect resources, prompts, and tools from all servers in one go using fresh sessions"""
        prompts: list[Prompt] = []
        resources: list[Resource] = []
        tools: list[Tool] = []
        server_by_tool_name = {}

        for server_idx, (server_name, server_config) in enumerate(self.server_configs.items()):
            async with get_client(server_config, self.timeout) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    try:
                        prompts.extend((await session.list_prompts()).prompts)
                    except Exception:
                        logger.exception(f"Error listing prompts for server {server_idx}")

                    try:
                        resources.extend((await session.list_resources()).resources)
                    except Exception:
                        logger.exception(f"Error listing resources for server {server_idx}")

                    try:
                        tool_list = (await session.list_tools()).tools
                        server_by_tool_name.update({tool.name: server_idx for tool in tool_list})
                        tools.extend(tool_list)
                    except Exception:
                        logger.exception(f"Error listing tools for server {server_idx}")

        self.prompts = prompts
        self.resources = resources
        self.tools = self._convert_mcp_tools_to_openai_format(tools)
        self.server_by_tool_name = server_by_tool_name
        self.message_history = [{"role": "system", "content": SYSTEM_PROMPT_AGENT}]

    async def _call_tool(self, tool_name: str, arguments: str, tool_call_id: str) -> tuple[str, str]:
        """Create a new session just for this tool call and close it properly"""
        logger.debug(f"Calling tool {tool_name} with arguments: {arguments}")
        server_idx = self.server_by_tool_name[tool_name]
        server_name = list(self.server_configs.keys())[server_idx]
        server_config = self.server_configs[server_name]

        async with get_client(server_config, self.timeout) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    response = await session.call_tool(tool_name, arguments=json.loads(arguments))
                    if response.isError:
                        return tool_call_id, f"Error calling tool {tool_name} with arguments: {arguments}: {response}"
                    results = []
                    for content in response.content:
                        if isinstance(content, TextContent):
                            results.append(content.text)
                        elif isinstance(content, ImageContent):
                            results.append(content.image)
                        elif isinstance(content, EmbeddedResource):
                            results.append(content.resource)
                    return tool_call_id, "\n".join(results)
                except Exception as e:
                    logger.exception(f"Error calling tool {tool_name} with arguments: {arguments}")
                    return tool_call_id, f"Error calling tool {tool_name} with arguments: {arguments}: {e}"

    async def talk(self, user_message: Optional[str] = None) -> str:
        if not self.tools:
            await self._collect_resources()

        if user_message:
            self.message_history.append({"role": "user", "content": user_message})

        # Initial LLM call
        response = await litellm.acompletion(
            messages=self.message_history,
            tools=self.tools,
            **self.llm_config
        )
        logger.debug(f"Agent response [1]: {response.choices[0]}")
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Loop to handle multiple rounds of tool calls
        while tool_calls:
            self.message_history.append(response_message.json())

            for tool_call in tool_calls:
                tool_call_id = tool_call.id
                tool_call_name = tool_call.function.name
                tool_call_args = tool_call.function.arguments
                logger.debug(f"Tool call: {tool_call_name} with args: {tool_call_args}")

                call_id, tool_call_result = await self._call_tool(tool_call_name, tool_call_args, tool_call_id)

                # Add tool response with the corresponding tool_call_id
                self.message_history.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_call_name,
                    "content": tool_call_result
                })

            # Get next response after tool calls
            response = await litellm.acompletion(
                messages=self.message_history,
                tools=self.tools,
                **self.llm_config
            )
            response_message = response.choices[0].message
            logger.debug(f"Agent response (after tool calls): {response_message}")
            tool_calls = response_message.tool_calls

        # Add the final assistant message to history
        self.message_history.append({"role": "assistant", "content": response_message.content})
        return response_message.content
