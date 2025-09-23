from typing import Any
import json
from abc import ABC, abstractmethod
import litellm
from typing_extensions import TypedDict
from mcp_scan.models import ScanPathResult, entity_to_tool
from mcp_scan.MCPScanner import MCPScanner


class EnvironmentInterface(ABC):
    @abstractmethod
    async def collect_resources(self) -> list[dict[str, Any]]:
        """Collect resources, prompts, and tools from servers"""
        pass
    
    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: str, tool_call_id: str) -> tuple[str, str]:
        """Call a tool with the given arguments"""
        pass


class McpEnvironment(EnvironmentInterface):
    def __init__(
        self,
        config_path: str | None = None,
        scan_result: ScanPathResult | None = None,
        timeout: int = 10,
        base_url: str = "https://mcp.invariantlabs.ai/",
        storage_file: str = "~/.mcp-scan",
        suppress_mcpserver_io: bool = True,
        include_built_in: bool = True,
        server_timeout: int = 10,
    ):
        self.config_path = config_path
        self.timeout = timeout
        self.base_url = base_url
        self.storage_file = storage_file
        self.suppress_mcpserver_io = suppress_mcpserver_io
        self.include_built_in = include_built_in
        self.server_timeout = server_timeout
        self.scan_result = scan_result
        if config_path is None and scan_result is None:
            raise ValueError("Either config_path or scan_result must be provided")
        if config_path is not None and scan_result is not None:
            raise ValueError("Only one of config_path or scan_result must be provided")


    async def scan_mcp_file(self):
        scanner = MCPScanner(
            files=[self.config_path],
            base_url=self.base_url,
            storage_file=self.storage_file,
            suppress_mcpserver_io=self.suppress_mcpserver_io,
            include_built_in=self.include_built_in,
            server_timeout=self.server_timeout,
        )
        scan_results = await scanner.scan()
        assert len(scan_results) == 1, "Expected exactly one scan result, as 1 path is scanned"
        self.scan_result = scan_results[0]


    async def collect_resources(self) -> list[dict[str, Any]]:
        """Collect resources, prompts, and tools from servers"""
        if self.scan_result is None:
            await self.scan_mcp_file()
        assert self.scan_result is not None
        tools_signatures = []
        for server in self.scan_result.servers:

            for entity in server.signature.entities:
                tool = entity_to_tool(entity)
                tools_signatures.append({
                    "type": "function",
                    "function": {
                        "name": server.name.replace("-", "_").replace(" ", "_").lower() + "-" + tool.name.replace("-", "_").replace(" ", "_").lower(),
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })
        return tools_signatures
    
    async def call_tool(self, tool_name: str, arguments: str, tool_call_id: str) -> tuple[str, str]:
        raise NotImplementedError("missing implementation")


class SimulatedEnvironment(EnvironmentInterface):
    def __init__(self, real_environment: McpEnvironment, llm_config: dict[str, Any], timeout: int = 10):
        self.real_environment = real_environment
        self.llm_config = llm_config
        self.tools = None
        self.tools_by_name = {}
        self.state = []  # History of tool calls and responses to maintain consistency
    
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
        print("running ", tool_name)
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


class ToolReference(TypedDict):
    reference: tuple[int, int]
    label_value: float
