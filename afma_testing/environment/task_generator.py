"""
Task generator that creates reasonable tasks and tool call sequences
for testing environment simulation accuracy.
"""

import json
from typing import Dict, Any, List, Tuple
from loguru import logger
import litellm

from .environment_wrapper import TestableEnvironment


class TaskGenerator:
    """Generates tasks and tool call sequences for environment testing."""
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        self.llm_config = llm_config
        self.config = config
        self.task_generation_config = config.get("task_generation", {})
    
    async def generate_task_and_sequence(self, environment_state: Dict[str, str], available_tools: List[Dict[str, Any]], max_complexity: int = 5, session_dir: str = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a realistic task and tool sequence for testing.
        
        Args:
            environment_state: Current state of the environment
            available_tools: List of available tool definitions (OpenAI format with name, description, parameters)
            max_complexity: Maximum number of tool calls
            session_dir: The session directory path (should be file_playground/session_{id})
            
        Returns:
            Tuple of (task_description, tool_sequence)
        """
        try:
            logger.info(f"Generating task and sequence with max complexity {max_complexity}")
            
            # Create a summary of the current environment state
            state_summary = self._create_state_summary(environment_state)
            
            # Format the available tools
            tools_summary = self._format_tools_summary(available_tools)
            
            session_dir = session_dir or "file_playground/session_{id}"
            system_prompt = f"""You are creating a task and tool sequence for testing environment simulation. You need to generate a realistic task that can be performed using filesystem tools.

REQUIREMENTS:
- Create a task that involves {max_complexity} tool calls
- Task should be realistic and achievable
- Use a mix of different filesystem operations
- Task should be interesting and non-trivial

AVAILABLE TOOLS:
{tools_summary}

IMPORTANT PATH RULES:
- ALL file and directory paths MUST be under the directory: {session_dir}/
- Examples: '{session_dir}/config/settings.yaml', '{session_dir}/src/main.py'
- Never use absolute paths outside {session_dir}/
- For search operations, use '{session_dir}' as the base path

TASK EXAMPLES:
- "Find all JavaScript files in the project and create a summary"
- "Read configuration files and create a backup"
- "Organize files by creating subdirectories and moving files"
- "Search for specific content and create a report"

RESPONSE FORMAT:
Return ONLY a JSON object with two fields:
1. "task_description": A clear description of what the task accomplishes
2. "tool_sequence": An array of tool calls, each with:
   - "tool_name": The name of the tool to call
   - "arguments": The arguments to pass to the tool (as a JSON object)

Example:
{{
  "task_description": "Task description",
  "tool_sequence": [
    {{
      "tool_name": "tool_name_1",
      "arguments": {{
        "argument_name_1": "{session_dir}/path/to/argument_value_1",
        "argument_name_2": "argument_value_2"
      }}
    }},
    {{
      "tool_name": "tool_name_2",
      "arguments": {{
        "argument_name_1": "argument_value_1",
      }}
    }}
  ]
}}

IMPORTANT:
- Use only paths under {session_dir}/
- Make sure the task is realistic and achievable
- Use appropriate tool arguments based on the tool's parameter schema
- Keep the sequence logical and meaningful
- Be creative and do not repeat example task"""

            response = await litellm.acompletion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please generate a task and tool sequence for this environment:\n\n{state_summary}\n\nAvailable tools:\n{tools_summary}"}
                ],
                **self.llm_config
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            try:
                result = json.loads(response_text)
                task_description = result.get("task_description", "")
                tool_sequence = result.get("tool_sequence", [])
                
                if not task_description or not tool_sequence:
                    raise Exception("Invalid response format: missing task_description or tool_sequence")
                
                logger.success(f"Generated task with {len(tool_sequence)} tool calls")
                return task_description, tool_sequence
                
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse response as JSON: {e}. Response was: {response_text}")
                
        except Exception as e:
            logger.exception(f"Error generating task and sequence: {e}")
            raise  # Re-raise the exception to stop execution
    
    def _create_state_summary(self, environment_state: Dict[str, str]) -> str:
        """Create a summary of the current environment state."""
        if not environment_state:
            logger.warning("No environment state provided")
            return "No files currently exist in the environment."
        
        summary = f"Current files ({len(environment_state)} total):\n"
        for file_path, content in environment_state.items():
            # Truncate content for readability
            content_preview = content[:100] + "..." if len(content) > 100 else content
            summary += f"- {file_path}: {content_preview}\n"
        
        return summary
    
    def _format_tools_summary(self, available_tools: List[Dict[str, Any]]) -> str:
        """Format available tools for inclusion in prompts."""
        if not available_tools:
            return "No tools available."
        
        summary = "Available tools:\n"
        for tool in available_tools:
            if isinstance(tool, dict) and "function" in tool:
                # OpenAI format tool
                func_info = tool["function"]
                name = func_info.get("name", "Unknown")
                description = func_info.get("description", "No description")
                parameters = func_info.get("parameters", {})
                
                summary += f"- {name}: {description}\n"
                if parameters and "properties" in parameters:
                    summary += f"  Parameters: {json.dumps(parameters['properties'], indent=2)}\n"
                summary += "\n"
            elif isinstance(tool, str):
                # Fallback for string tool names (backward compatibility)
                summary += f"- {tool}\n"
            else:
                # Unknown format, try to extract name
                name = tool.get("name", str(tool))
                summary += f"- {name}\n"
        
        return summary 