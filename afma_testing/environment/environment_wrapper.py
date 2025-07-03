"""
Environment wrapper classes that extend the base Environment and SimulatedEnvironment
with additional functionality for testing purposes.
"""

import json
from typing import Dict, Any, List, Optional
from loguru import logger
import litellm

from ..simulation.environment import Environment
from ..simulation.agent import SimulatedEnvironment


class TestableEnvironment(Environment):
    """Extended Environment class with get_state functionality for testing."""
    
    def __init__(self, mcp_config_path: str, timeout: int = 10, llm_config: Optional[Dict[str, Any]] = None, session_dir: Optional[str] = None):
        super().__init__(mcp_config_path, timeout)
        self.llm_config = llm_config or {}
        self.session_dir = session_dir or "."
    
    async def get_state(self) -> Dict[str, str]:
        """
        Get the current state of the filesystem environment.
        Returns a dictionary mapping file paths to their content.
        """
        try:
            # Ensure tools are collected first
            if not self.tools:
                await self.collect_resources()
            
            # Debug: Log available tools
            logger.info(f"Available tools: {list(self.server_by_tool_name.keys())}")
            
            # First, get the directory tree to understand the structure
            tree_response = await self.call_tool(
                "filesystem_directory_tree", 
                json.dumps({"path": self.session_dir}), 
                "get_tree"
            )
            
            if tree_response[1].startswith("Error"):
                raise Exception(f"Failed to get directory tree: {tree_response[1]}")
            
            # Parse the tree response to extract file paths
            try:
                tree_data = json.loads(tree_response[1])
                logger.debug(f"Directory tree response: {json.dumps(tree_data, indent=2)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse directory tree response as JSON: {e}. Response: {tree_response[1]}")
            
            file_paths = self._extract_file_paths(tree_data, self.session_dir)
            
            if not file_paths:
                logger.info("No files found in the environment")
                return {}
            
            logger.info(f"Found {len(file_paths)} files to read: {file_paths}")
            
            # Read all files at once
            read_response = await self.call_tool(
                "filesystem_read_multiple_files",
                json.dumps({"paths": file_paths}),
                "read_files"
            )
            
            if read_response[1].startswith("Error"):
                raise Exception(f"Failed to read files: {read_response[1]}")
            
            # Parse the custom response format from the MCP server
            # Format: file_path: content --- file_path: content
            response_text = read_response[1]
            state = {}
            
            # Split by the separator
            file_sections = response_text.split("---")
            
            for section in file_sections:
                section = section.strip()
                if not section:
                    continue
                
                # Find the first newline to separate path from content
                first_newline = section.find('\n')
                if first_newline == -1:
                    continue
                
                file_path = section[:first_newline].rstrip(':')
                content = section[first_newline + 1:].strip()
                
                # Use the file path as is (no prefix removal needed)
                state[file_path] = content
            
            logger.info(f"Retrieved state for {len(state)} files")
            return state
            
        except Exception as e:
            logger.exception(f"Error getting environment state: {e}")
            raise  # Re-raise the exception to stop execution
    
    def _extract_file_paths(self, tree_data: Any, base_path: str) -> List[str]:
        """Recursively extract all file paths from the directory tree."""
        file_paths = []
        
        logger.debug(f"_extract_file_paths called with base_path: '{base_path}', tree_data type: {type(tree_data)}")
        
        if isinstance(tree_data, dict):
            if tree_data.get("type") == "file":
                logger.debug(f"Found file: {tree_data.get('name')} at path: {base_path}")
                file_paths.append(base_path)
            elif tree_data.get("type") == "directory":
                children = tree_data.get("children", [])
                logger.debug(f"Found directory: {tree_data.get('name')} with {len(children)} children")
                for child in children:
                    child_name = child.get("name", "")
                    # Build the path for this child by appending to the current directory path
                    if base_path.endswith("/"):
                        child_path = f"{base_path}{tree_data.get('name')}/{child_name}"
                    else:
                        child_path = f"{base_path}/{tree_data.get('name')}/{child_name}"
                    logger.debug(f"Processing child '{child_name}' with path: '{child_path}'")
                    file_paths.extend(self._extract_file_paths(child, child_path))
        elif isinstance(tree_data, list):
            logger.debug(f"Processing list with {len(tree_data)} items")
            for item in tree_data:
                file_paths.extend(self._extract_file_paths(item, base_path))
        
        logger.debug(f"Returning file_paths: {file_paths}")
        return file_paths


class TestableSimulatedEnvironment(SimulatedEnvironment):
    """Extended SimulatedEnvironment class with get_state functionality for testing."""
    
    def __init__(self, mcp_config_path: str, llm_config: dict[str, Any], timeout: int = 10, 
                 personality: Optional[str] = None, environment_expectations: Optional[str] = None, session_dir: Optional[str] = None):
        super().__init__(mcp_config_path, llm_config, timeout, personality, environment_expectations)
        self.session_dir = session_dir or "."
    
    async def get_state(self) -> Dict[str, str]:
        """
        Get the current state of the simulated filesystem environment using LLM.
        Returns a dictionary mapping file paths to their content.
        """
        try:
            # Ensure tools are collected first
            if not self.tools:
                await self.collect_resources()
            
            # Create a prompt for the LLM to describe the current state
            system_prompt = f"""You are simulating a filesystem environment. Based on the history of tool calls and their results, you need to describe the current state of the filesystem.

AVAILABLE TOOLS:
{self._format_tools_for_state()}

TOOL CALL HISTORY:
{json.dumps(self.state, indent=2)}

TASK: Describe the current state of the filesystem after all the tool calls have been executed. Focus on:
1. All files that exist and their content
2. Directory structure
3. Only include files and directories within the session directory: {self.session_dir}

RESPONSE FORMAT:
Return ONLY a JSON object where:
- Keys are file paths (FULL paths including the session directory prefix: {self.session_dir}/)
- Values are the file contents as strings
- Include all files that currently exist in the filesystem
- Do not include directories (only files with content)

Example:
{{
  "{self.session_dir}/config.txt": "server_port=8080\\ndebug=true",
  "{self.session_dir}/src/main.py": "def main():\\n    print('Hello World')",
  "{self.session_dir}/src/readme.md": "# Project Documentation\\n\\nThis is a test project."
}}

IMPORTANT:
- Only include files that actually exist after the tool calls
- Use realistic file contents based on the tool call history
- Do not include files that were deleted or never created
- Focus on the final state after all operations are complete
- ALWAYS include the full session directory prefix: {self.session_dir}/ in all file paths
- This ensures consistency with the real environment state format"""

            response = await litellm.acompletion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please describe the current filesystem state."}
                ],
                **self.llm_config
            )
            
            state_text = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                state = json.loads(state_text)
                logger.info(f"Retrieved simulated state for {len(state)} files")
                return state
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse LLM response as JSON: {e}. Response was: {state_text}")
                
        except Exception as e:
            logger.exception(f"Error getting simulated environment state: {e}")
            raise  # Re-raise the exception to stop execution
    
    def _format_tools_for_state(self) -> str:
        """Format tool information for the state generation prompt."""
        if not self.tools_by_name:
            return "No tools available"
        
        formatted = []
        for tool_name, tool_info in self.tools_by_name.items():
            formatted.append(f"- {tool_name}: {tool_info.get('description', 'No description')}")
        
        return "\n".join(formatted) 