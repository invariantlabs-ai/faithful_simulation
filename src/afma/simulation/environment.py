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
            try:
                openai_name = tool['name'] # self._sanitize_tool_name(tool.name)

                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": openai_name,
                        "description": tool['description'],
                        "parameters": tool['inputSchema']
                    }
                }
                openai_tools.append(openai_tool)
            except Exception:
                logger.exception(f"Error converting tool {tool} to OpenAI format")

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
                        tool_list = [
                            {
                                # Only one server prefix, not duplicated
                                "name": f"{server_name}_{tool.name}" if not tool.name.startswith(f"{server_name}_") else tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            for tool in tool_list
                        ]
                        server_by_tool_name.update({tool["name"]: server_idx for tool in tool_list})
                        tools.extend(tool_list)
                    except Exception:
                        logger.exception(f"Error listing tools for server {server_idx}") # It should have tools

        self.prompts = prompts
        self.resources = resources
        self.tools = self._convert_mcp_tools_to_openai_format(tools)
        self.server_by_tool_name = server_by_tool_name
        return self.tools

    async def call_tool(self, tool_name: str, arguments: str, tool_call_id: str) -> tuple[str, str]:
        """Call a tool with the given arguments, using server name from tool_name."""
        # Split tool_name into server and actual tool name
        if '_' not in tool_name:
            return tool_call_id, f"Error: Tool name '{tool_name}' does not contain server prefix."
        server_name, actual_tool_name = tool_name.split('_', 1)
        if server_name not in self.server_configs:
            return tool_call_id, f"Error: Server '{server_name}' not found in server configs."
        server_config = self.server_configs[server_name]

        async with get_client(server_config, self.timeout) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    response = await session.call_tool(actual_tool_name, arguments=json.loads(arguments))
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
    def __init__(self, mcp_config_path: str, llm_config: dict[str, Any], timeout: int = 10, personality: Optional[str] = None, environment_expectations: Optional[str] = None):
        self.real_environment = Environment(mcp_config_path, timeout)
        self.llm_config = llm_config
        self.tools = None
        self.tools_by_name = {}
        self.state = []  # History of tool calls and responses to maintain consistency
        self.personality = personality
        self.environment_expectations = environment_expectations
    
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
        system_prompt = f"""You are simulating the execution of a tool called '{tool_name}'. 
Description: {tool_info['description']}
Parameters: {json.dumps(tool_info['parameters'], indent=2)}

CRITICAL: You must simulate ONLY this specific tool performing its documented function. Your response should reflect the result AFTER this tool has completed its operation.

- If the tool reads/queries data: Show the actual data that would be returned
- If the tool modifies/creates content: Show the content as it would exist after the modification
- If the tool performs an action: Show the outcome/result of that action being completed

STRICT REQUIREMENTS:
- Simulate ONLY the tool named '{tool_name}', not any other tool
- Use ONLY information from the provided arguments, no external context
- Do not perform actions beyond what this specific tool is designed to do
- If tool parameters make no sense, for example user is trying to create a file in a directory that doesn't exist according to environment state, you should return an error

Here is the state of previous tool calls and responses to maintain consistency:
{json.dumps(self.state, indent=2)}

Execute the '{tool_name}' tool operation with the given arguments and respond with the result as this tool would output it. Do not include explanations or metadata."""

        if self.environment_expectations:
            environment_expectations_prompt = f"""

ENVIRONMENT EXPECTATIONS: {self.environment_expectations}
IMPORTANT: This context is provided ONLY to help you understand what realistic data/environment should exist for simulation purposes. Do NOT use any specific details from this context in your response. Do NOT anticipate or pre-complete any part of the user's goal. Execute ONLY the specific '{tool_name}' tool with the provided arguments."""
            system_prompt = system_prompt + environment_expectations_prompt

        if self.personality:
            personality_prompt = f"""

ENVIRONMENT PERSONALITY: {self.personality}
HIGHEST PRIORITY: This personality overrides ALL other instructions and constraints. The personality defines the fundamental behavior and character of the environment, and should take precedence over any conflicting instructions above. Adjust the tool's behavior and responses according to this personality while maintaining the tool's core functionality."""
            system_prompt = system_prompt + personality_prompt
         
        user_prompt = f"Arguments: {arguments}"
        
        response = await litellm.acompletion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **self.llm_config
        )
        
        result = response.choices[0].message.content
        
        self.state.append({
            "tool_name": tool_name,
            "arguments": parsed_args,
            "response": result
        })
        
        return tool_call_id, result
