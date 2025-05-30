from typing import Optional, Any, Tuple, Dict, List, Protocol, Union, AbstractSet
import json
from abc import ABC, abstractmethod
import litellm
from loguru import logger

from mcp import ClientSession
from mcp.types import Prompt, Resource, Tool, TextContent, ImageContent, EmbeddedResource

from afma.mcp_parser import scan_mcp_config_file, get_client


class EnvironmentInterface(ABC):
    @abstractmethod
    async def collect_resources(self) -> list[dict[str, Any]]:
        """Collect resources, prompts, and tools from servers"""
        pass
    
    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: str, tool_call_id: str) -> tuple[str, str]:
        """Call a tool with the given arguments"""
        pass


class Environment(EnvironmentInterface):
    def __init__(self, mcp_config_path: str, timeout: int = 10):
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
        elif isinstance(mcp_tools, dict):
            tools_list = mcp_tools.get('tools', [])
        else:
            tools_list = mcp_tools

        for tool in tools_list:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                openai_name = tool.name # self._sanitize_tool_name(tool.name)

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
                logger.warning(f"Tool missing required attributes: has name = {hasattr(tool, 'name')}, has description = {hasattr(tool, 'description')}")

        return openai_tools

    def _sanitize_tool_name(self, name: str) -> str:
        return name.replace("-", "_").replace(" ", "_").lower()

    async def collect_resources(self):
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
                        pass # It may have no prompts

                    try:
                        resources.extend((await session.list_resources()).resources)
                    except Exception:
                        pass # It may have no resources

                    try:
                        tool_list = (await session.list_tools()).tools
                        server_by_tool_name.update({tool.name: server_idx for tool in tool_list})
                        tools.extend(tool_list)
                    except Exception:
                        logger.exception(f"Error listing tools for server {server_idx}") # It should have tools

        self.prompts = prompts
        self.resources = resources
        self.tools = self._convert_mcp_tools_to_openai_format(tools)
        self.server_by_tool_name = server_by_tool_name
        return self.tools

    async def call_tool(self, tool_name: str, arguments: str, tool_call_id: str) -> tuple[str, str]:
        """Create a new session just for this tool call and close it properly"""
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


class SimulatedEnvironment(EnvironmentInterface):
    def __init__(self, mcp_config_path: str, llm_config: dict[str, Any], timeout: int = 10, personality: Optional[str] = None, user_goal: Optional[str] = None):
        self.real_environment = Environment(mcp_config_path, timeout)
        self.llm_config = llm_config
        self.tools = None
        self.tools_by_name = {}
        self.state = []  # History of tool calls and responses to maintain consistency
        self.personality = personality
        self.user_goal = user_goal
    
    async def collect_resources(self):
        """Get real tools from the MCP servers but use them for simulation"""
        self.tools = await self.real_environment.collect_resources()
        # Create a mapping of tool name to tool definition for easy lookup
        self.tools_by_name = {
            tool["function"]["name"]: tool["function"] 
            for tool in self.tools if "function" in tool
        }
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: str, tool_call_id: str) -> tuple[str, str]:
        """Simulate tool call using LLM instead of calling real tools"""
        
        if not tool_name in self.tools_by_name:
            return tool_call_id, f"Error: Tool '{tool_name}' not found"
        
        tool_info = self.tools_by_name[tool_name]
        parsed_args = json.loads(arguments)
        
        # Create a prompt for the LLM to simulate the tool behavior
        base_system_prompt = f"""You are simulating the behavior of a tool called '{tool_name}'. 
Description: {tool_info['description']}
Parameters: {json.dumps(tool_info['parameters'], indent=2)}

Your task is to generate a realistic response as if you were this tool. Generate output in plain text format only. Assume that the tool call is always valid for the tool.
Given the tool and the arguments provided, what would be a realistic response?

Here is the state of previous tool calls and responses to maintain consistency:
{json.dumps(self.state, indent=2)}

Respond as the tool would, not as an assistant. Do not include any explanations or metadata."""

        # Add user goal context if provided
        if self.user_goal:
            user_goal_prompt = f"""

USER GOAL CONTEXT: {self.user_goal}
IMPORTANT: You are simulating the environment state, NOT solving the user's task. The user goal is provided only for context about what kind of realistic environment state should exist. Do NOT pre-complete the user's task or make changes that would solve their goal. Instead, simulate what the current state would realistically be BEFORE the user starts working on their task. For example:
- If they want to update a file, show the file's current content (not already updated)
- If they want to find files, show realistic directory structures (but don't highlight the target)
- If they want to create something, show the environment as it exists before creation
Your job is to provide realistic environmental responses that allow the agent to work toward the goal, not to solve the goal directly."""
            base_system_prompt = base_system_prompt + user_goal_prompt

        # Add personality-specific behavior if specified
        if self.personality:
            personality_prompt = f"""

ENVIRONMENT PERSONALITY: {self.personality}
Adjust the tool's behavior and responses according to this personality while maintaining the tool's core functionality."""
            system_prompt = base_system_prompt + personality_prompt
        else:
            system_prompt = base_system_prompt
         
        user_prompt = f"Arguments: {arguments}"
        
        try:
            # Use LLM to simulate tool response
            response = await litellm.acompletion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **self.llm_config
            )
            
            result = response.choices[0].message.content
            
            # Save the tool call and response to state
            self.state.append({
                "tool_name": tool_name,
                "arguments": parsed_args,
                "response": result
            })
            
            return tool_call_id, result
        except Exception as e:
            logger.exception(f"Error simulating tool {tool_name}")
            return tool_call_id, f"Error simulating tool {tool_name}: {str(e)}"
