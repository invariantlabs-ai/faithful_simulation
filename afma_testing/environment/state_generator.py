"""
Environment state generator that creates interesting filesystem states
of varying difficulty levels for testing purposes.
"""

import json
import random
from typing import Dict, Any, List
from loguru import logger
import litellm

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
Return ONLY a JSON object where:
- Keys are file paths (relative paths, but always starting with {session_dir}/)
- Values are the file contents as strings

Example:
{{
  "{session_dir}/src/README.md": "# Project Overview\n\nThis is a test.",
  "{session_dir}/src/main.py": "def main():\n    print('Hello World')"
  "{session_dir}/config/main/info.yaml": "app:\n  name: SampleApp\n  version: 1.0.0\nlogging:\n  level: INFO\n  file: logs/app.log\nfeatures:\n  enable_feature_x: true\n  max_connections: 10"
}}

IMPORTANT:
- Create realistic file structures (use subdirectories when appropriate)
- Ensure file content matches the file type
- Keep content concise but meaningful
- Avoid creating files that would conflict with each other
- Make the environment interesting and varied
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
        try:
            # Try to parse as JSON
            file_states = json.loads(description)
            
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
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse state description as JSON: {e}. Description was: {description}")
    
    async def _create_files_in_environment(self, file_states: Dict[str, str], environment: TestableEnvironment, session_dir: str):
        """Create the files in the real environment."""
        
        # First, collect all unique directories that need to be created
        directories = set()
        for file_path in file_states.keys():
            if '/' in file_path:
                dir_path = '/'.join(file_path.split('/')[:-1])
                directories.add(dir_path)
        
        # Create directories first
        for dir_path in sorted(directories):
            try:
                # Ensure directory is under session_dir
                if not dir_path.startswith(session_dir):
                    dir_path = f"{session_dir}/" + dir_path.lstrip("/")
                logger.debug(f"Creating directory: {dir_path}")
                result = await environment.call_tool(
                    "filesystem_create_directory",
                    json.dumps({"path": dir_path}),
                    f"create_dir_{dir_path}"
                )
                logger.debug(f"Directory creation result: {result}")
                if result[1].startswith("Error"):
                    logger.error(f"Failed to create directory {dir_path}: {result[1]}")
                    raise Exception(f"Failed to create directory {dir_path}: {result[1]}")
                logger.debug(f"Created directory: {dir_path}")
            except Exception as e:
                raise Exception(f"Failed to create directory {dir_path}: {e}")
        
        # Create files
        for file_path, content in file_states.items():
            try:
                # Ensure file is under session_dir
                if not file_path.startswith(session_dir):
                    file_path = f"{session_dir}/" + file_path.lstrip("/")
                logger.debug(f"Creating file: {file_path} with content length: {len(content)}")
                result = await environment.call_tool(
                    "filesystem_write_file",
                    json.dumps({
                        "path": file_path,
                        "content": content
                    }),
                    f"create_file_{file_path}"
                )
                logger.debug(f"File creation result: {result}")
                if result[1].startswith("Error"):
                    logger.error(f"Failed to create file {file_path}: {result[1]}")
                    raise Exception(f"Failed to create file {file_path}: {result[1]}")
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