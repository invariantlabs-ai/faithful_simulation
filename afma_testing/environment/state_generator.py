"""
Environment state generator that creates interesting filesystem states
of varying difficulty levels for testing purposes.
"""

import json
import pyjson5
import random
from typing import Dict, Any, List
from loguru import logger
import litellm
from pathlib import Path

from .environment_wrapper import TestableEnvironment


class EnvironmentStateGenerator:
    """Generates interesting filesystem environment states for testing."""
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        self.llm_config = llm_config
        self.config = config
        self.state_generation_config = config.get("state_generation", {})
    
    async def generate_environment_state(self, difficulty: int, real_environment: TestableEnvironment, session_dir: str = None) -> Dict[str, str]:
        """
        Generate an environment state with the specified difficulty level.
        
        Args:
            difficulty: Number of files to create
            real_environment: The real environment to use for creating the state
            session_dir: The session directory path (should be file_playground/session_{id})
            
        Returns:
            Dictionary mapping file paths to content
        """
        try:
            logger.info(f"Generating environment state with difficulty {difficulty}")
            
            # Ensure tools are collected first
            if not real_environment.tools:
                await real_environment.collect_resources()
            
            # Use the session_dir from the environment if not provided
            session_dir = session_dir or getattr(real_environment, 'session_dir', None) or '.'
            
            # Generate the state description using LLM
            state_description = await self._generate_state_description(difficulty, session_dir)
            
            # Parse the state description
            file_states = self._parse_state_description(state_description)
            
            if not file_states:
                raise Exception("Failed to generate valid file states")
            
            # Create the files in the real environment
            await self._create_files_in_environment(file_states, real_environment, session_dir)
            
            logger.success(f"Successfully created environment state with {len(file_states)} files")
            return file_states
            
        except Exception as e:
            logger.exception(f"Error generating environment state: {e}")
            raise  # Re-raise the exception to stop execution
    
    async def _generate_state_description(self, difficulty: int, session_dir: str) -> str:
        """Generate a description of the environment state using LLM."""
        
        max_files = difficulty  # Use difficulty directly as file count
        max_content_length = self.config.get("test", {}).get("max_file_content_length", 500)
        file_types = self.state_generation_config.get("file_types", ["txt", "md", "py", "js", "json", "yaml"])
        content_themes = self.state_generation_config.get("content_themes", ["documentation", "code", "data", "configuration"])
        
        system_prompt = f"""You are creating a filesystem environment for testing purposes. You need to generate a realistic set of files with appropriate content.

REQUIREMENTS:
- Create exactly {max_files} files
- File content should be realistic and not exceed {max_content_length} characters
- Use a mix of file types: {', '.join(file_types)}
- Content themes: {', '.join(content_themes)}
- ALL file and directory paths MUST be under the directory: {session_dir}/
- For example: {session_dir}/docs/README.md, {session_dir}/src/main.py, etc.

AVAILABLE FILE TYPES:
- txt: Plain text files (documentation, notes, logs)
- md: Markdown files (documentation, readme files)
- py: Python files (scripts, modules)
- js: JavaScript files (scripts, modules)
- json: JSON files (configuration, data)
- yaml: YAML files (configuration, data)

CONTENT THEMES:
- documentation: README files, API docs, user guides
- code: Scripts, modules, functions
- data: Configuration files, data files, logs
- configuration: Settings, config files, environment files

RESPONSE FORMAT:
Return ONLY a valid JSON object that can be parsed by standard JSON parsers.
- Keys: file paths (relative paths, always starting with {session_dir}/)
- Values: file contents as properly escaped JSON strings
- All paths must be valid filesystem paths (no invalid characters like |, <, >, etc.)
- Use forward slashes (/) for path separators
- Ensure proper JSON escaping for special characters in content

JSON ESCAPING RULES:
- Use \\n for actual newlines in file content
- Use \\\\ for literal backslash characters
- Use \\" for literal quote characters
- Use \\t for tab characters
- All string values must be properly quoted and escaped

Example:
{{
  "{session_dir}/src/README.md": "# Project Overview\\n\\nThis is a test project with multiple lines.",
  "{session_dir}/src/main.py": "def main():\\n    print(\\"Hello World\\")\\n    return 0",
  "{session_dir}/config/app.yaml": "app:\\n  name: SampleApp\\n  version: 1.0.0\\nlogging:\\n  level: INFO\\n  file: logs/app.log"
}}

CRITICAL REQUIREMENTS:
- Output must be valid, parseable JSON (test with any JSON parser)
- All file paths must be valid and start with {session_dir}/
- No missing commas, quotes, or brackets
- Proper escaping of all special characters in file content
- Create realistic file structures (use subdirectories when appropriate)
- Ensure file content matches the file type and is meaningful
- Do NOT include any files or directories outside of {session_dir}/"""

        response = await litellm.acompletion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate a filesystem environment with {max_files} files."}
            ],
            **self.llm_config
        )
        
        return response.choices[0].message.content.strip()
    
    def _parse_state_description(self, description: str) -> Dict[str, str]:
        """Parse the LLM-generated state description into a file dictionary."""
        # Clean up common formatting issues from LLM responses
        description = description.strip()
        # Remove markdown code fences
        description = description.replace('```json', '').replace('```', '')
        # Remove any leading/trailing whitespace again after cleanup
        description = description.strip()
        
        # Try multiple parsing approaches, starting with the most flexible
        parsing_errors = []
        
        # First try pyjson5 (most flexible)
        try:
            file_states = pyjson5.loads(description)
            logger.debug("Successfully parsed JSON using pyjson5")
        except Exception as e:
            parsing_errors.append(f"pyjson5: {e}")
            logger.debug(f"pyjson5 parsing failed: {e}")
            
            # Fallback to standard json
            try:
                file_states = json.loads(description)
                logger.debug("Successfully parsed JSON using standard json")
            except json.JSONDecodeError as e:
                parsing_errors.append(f"json: {e}")
                error_msg = f"Failed to parse JSON with both pyjson5 and standard json.\nErrors: {'; '.join(parsing_errors)}\nDescription was:\n===\n{description}\n==="
                raise Exception(error_msg)
        
        # Validate the structure
        if not isinstance(file_states, dict):
            raise Exception("Generated state is not a dictionary")
        
        # Validate each file
        valid_files = {}
        for file_path, content in file_states.items():
            if isinstance(file_path, str) and isinstance(content, str):
                valid_files[file_path] = content
            else:
                logger.warning(f"Invalid file entry: {file_path}")
        
        return valid_files
    
    async def _create_files_in_environment(self, file_states: Dict[str, str], environment: TestableEnvironment, session_dir: str):
        """Create the files in the real environment using standard Python file operations."""
        
        # Create files directly using pathlib - this will automatically create parent directories
        for file_path, content in file_states.items():
            try:
                # Ensure file is under session_dir
                if not file_path.startswith(session_dir):
                    file_path = f"{session_dir}/" + file_path.lstrip("/")
                
                logger.debug(f"Creating file: {file_path} with content length: {len(content)}")
                
                # Convert to Path object
                file_path_obj = Path(file_path)
                
                # Create parent directories if they don't exist
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the file content
                file_path_obj.write_text(content, encoding='utf-8')
                
                logger.debug(f"Created file: {file_path}")
            except Exception as e:
                raise Exception(f"Failed to create file {file_path}: {e}")
    
    async def cleanup_environment(self, environment: TestableEnvironment):
        """Clean up the test environment by removing all files."""
        try:
            # Get the current state to see what files exist
            current_state = await environment.get_state()
            
            if not current_state:
                logger.info("No files to clean up")
                return
            
            # For now, we'll just log what would be cleaned up
            # In a real implementation, you might want to add delete_file tool
            logger.info(f"Would clean up {len(current_state)} files: {list(current_state.keys())}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise  # Re-raise the exception to stop execution 