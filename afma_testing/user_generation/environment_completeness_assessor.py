import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from loguru import logger
import litellm


@dataclass
class EnvironmentCompletenessResult:
    """Result of environment completeness assessment."""
    score: float  # number_of_issues / n_tools, where higher means more issues
    details: Dict[str, Any]
    missing_elements: List[str]
    reasoning: str
    error: Optional[str] = None


class EnvironmentCompletenessAssessor:
    """Assesses whether environment expectations are complete for achieving user goals."""
    
    def __init__(self, llm_config: Dict[str, Any], concurrency: int = 5):
        """
        Initialize the environment completeness assessor.
        
        Args:
            llm_config: LLM configuration for assessment
            concurrency: Maximum concurrent assessments
        """
        self.llm_config = llm_config
        self.semaphore = asyncio.Semaphore(concurrency)
        
    async def assess_completeness(self, user_goal: str, environment_expectations: str, tool_sequence: List[Dict[str, Any]]) -> EnvironmentCompletenessResult:
        """
        Assess if environment expectations are complete for achieving the user goal.
        
        Args:
            user_goal: The user's goal description
            environment_expectations: Current environment expectations
            tool_sequence: The sequence of tools that need to be executed
            
        Returns:
            EnvironmentCompletenessResult with assessment details
        """
        async with self.semaphore:
            try:
                system_prompt = """You are an expert at analyzing whether environment expectations contain all necessary information to execute a specific tool sequence. Your task is to identify ALL missing elements that would prevent successful execution.

CRITICAL EVALUATION CRITERIA:
1. **Authentication & Permissions**: Are all required credentials, tokens, or permissions mentioned?
2. **Tool Availability**: Are the specific tools/APIs needed for each step in the sequence available?
3. **Data/File Existence**: Are all required files, directories, or data sources explicitly mentioned?
4. **Network/Connectivity**: Are external service connections (GitHub, APIs, etc.) assumed but not stated?
5. **Environment Setup**: Are required software, configurations, or environment variables mentioned?
6. **Error Handling**: Are potential failure scenarios or error conditions addressed?
7. **Dependencies**: Are all prerequisites for each tool in the sequence covered?

RESPONSE FORMAT:
You must respond with a JSON object in this exact format:
{
    "reasoning": "<detailed explanation of your assessment>",
    "missing_elements": ["<list of specific missing elements>"]
}

IMPORTANT: Be thorough and identify ALL missing elements. Each missing element should be a specific, actionable item that would prevent execution. Don't be lenient - if something is assumed but not explicitly stated, include it as missing."""

                user_prompt = f"""USER GOAL:
{user_goal}

CURRENT ENVIRONMENT EXPECTATIONS:
{environment_expectations}

REQUIRED TOOL SEQUENCE:
{self._format_tool_sequence_info(tool_sequence)}

Please identify ALL missing elements in the environment expectations that would prevent successful execution of this tool sequence."""

                response = await litellm.acompletion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **self.llm_config
                )
                
                response_text = response["choices"][0]["message"]["content"].strip()
                
                # Parse JSON response
                try:
                    import json
                    result_data = json.loads(response_text)
                    
                    missing_elements = result_data.get("missing_elements", [])
                    reasoning = result_data.get("reasoning", "No reasoning provided")
                    
                    # Calculate score as number_of_issues / n_tools
                    score = len(missing_elements) / len(tool_sequence) if tool_sequence else 0.0
                    
                    return EnvironmentCompletenessResult(
                        score=score,
                        details={
                            "raw_response": response_text,
                            "n_tools": len(tool_sequence),
                            "n_missing_elements": len(missing_elements)
                        },
                        missing_elements=missing_elements,
                        reasoning=reasoning
                    )
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    # Fallback: try to extract missing elements from text
                    return self._parse_fallback_response(response_text, tool_sequence)
                    
            except Exception as e:
                logger.error(f"Error in environment completeness assessment: {e}")
                return EnvironmentCompletenessResult(
                    score=len(tool_sequence) if tool_sequence else 0.0,  # Assume all tools have issues if error
                    details={"error": str(e)},
                    missing_elements=[],
                    reasoning="Assessment failed due to error",
                    error=str(e)
                )
    
    def _format_tool_sequence_info(self, tool_sequence: List[Dict[str, Any]]) -> str:
        """Format tool sequence information for assessment."""
        formatted_info = ""
        for i, tool in enumerate(tool_sequence):
            name = tool["name"]
            description = tool["description"]
            formatted_info += f"{i+1}. {name}: {description}\n"
            
            # Add parameter information if available
            if "inputSchema" in tool and "properties" in tool["inputSchema"]:
                properties = tool["inputSchema"]["properties"]
                required_params = tool["inputSchema"].get("required", [])
                
                if properties:
                    formatted_info += f"   Parameters:\n"
                    for param_name, param_info in properties.items():
                        param_description = param_info.get("description", "")
                        is_required = param_name in required_params
                        required_label = " (required)" if is_required else " (optional)"
                        
                        if param_description:
                            formatted_info += f"     - {param_name}{required_label}: {param_description}\n"
                        else:
                            formatted_info += f"     - {param_name}{required_label}\n"
            formatted_info += "\n"
        
        return formatted_info.strip()
    
    def _parse_fallback_response(self, response_text: str, tool_sequence: List[Dict[str, Any]]) -> EnvironmentCompletenessResult:
        """Parse response when JSON parsing fails."""
        # Try to extract missing elements from text
        import re
        
        # Look for missing elements patterns
        missing_elements_patterns = [
            r'missing_elements["\s]*:["\s]*\[(["\s]*[^"]*["\s]*)*\]',
            r'Missing elements: ["\s]*\[(["\s]*[^"]*["\s]*)*\]',
            r'Missing elements: ["\s]*([^,]+)',
            r'Missing elements: ["\s]*([^,]+)'
        ]
        
        missing_elements = []
        for pattern in missing_elements_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                elements = match.group(1).split(",") if match.group(1) else []
                missing_elements = [element.strip() for element in elements]
                break
        
        # Calculate score as number_of_issues / n_tools
        score = len(missing_elements) / len(tool_sequence) if tool_sequence else 0.0
        
        return EnvironmentCompletenessResult(
            score=score,
            details={
                "raw_response": response_text,
                "parsed_with_fallback": True,
                "n_tools": len(tool_sequence),
                "n_missing_elements": len(missing_elements)
            },
            missing_elements=missing_elements,
            reasoning="Response parsed with fallback method"
        )
    
    async def assess_batch(self, user_data: List[Dict[str, Any]]) -> List[EnvironmentCompletenessResult]:
        """
        Assess environment completeness for a batch of users.
        
        Args:
            user_data: List of user data dictionaries with 'user_goal', 'environment_expectations', and 'source'
            
        Returns:
            List of EnvironmentCompletenessResult objects
        """
        tasks = []
        for user in user_data:
            task = self.assess_completeness(
                user["user_goal"], 
                user["environment_expectations"],
                user["source"]
            )
            tasks.append(task)
        
        logger.info(f"Assessing environment completeness for {len(tasks)} users")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error assessing user {i}: {result}")
                processed_results.append(EnvironmentCompletenessResult(
                    score=len(user_data[i]["source"]) if user_data[i]["source"] else 0.0,  # Assume all tools have issues if error
                    details={"error": str(result)},
                    missing_elements=[],
                    reasoning="Assessment failed due to exception",
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        logger.success(f"Completed environment completeness assessment for {len(processed_results)} users")
        return processed_results 