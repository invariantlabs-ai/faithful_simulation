import asyncio
import json
import pyjson5
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from loguru import logger
import litellm


@dataclass
class GoalUnambiguityResult:
    """Result of goal unambiguity assessment."""
    score: float  # number_of_issues / n_tools, where higher means more issues
    details: Dict[str, Any]
    missing_information: List[str]
    reasoning: str
    error: Optional[str] = None


class GoalUnambiguityAssessor:
    """Assesses whether user goals contain all necessary information for the original tool sequence."""
    
    def __init__(self, llm_config: Dict[str, Any], concurrency: int = 5):
        """
        Initialize the goal unambiguity assessor.
        
        Args:
            llm_config: LLM configuration for assessment
            concurrency: Maximum concurrent assessments
        """
        self.llm_config = llm_config
        self.semaphore = asyncio.Semaphore(concurrency)
    
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
        
    async def assess_unambiguity(self, 
                                user_goal: str, 
                                environment_expectations: str,
                                tool_sequence: List[Dict[str, Any]]) -> GoalUnambiguityResult:
        """
        Assess if user goal and environment expectations contain all necessary information for the tool sequence.
        
        Args:
            user_goal: The user's goal description
            environment_expectations: Environment expectations
            tool_sequence: Original tool sequence with descriptions and parameters
            
        Returns:
            GoalUnambiguityResult with assessment details
        """
        async with self.semaphore:
            try:
                tool_info = self._format_tool_sequence_info(tool_sequence)
                
                system_prompt = """You are an expert at analyzing whether user goals and environment expectations contain all necessary information to execute a specific tool sequence. Your task is to identify ALL missing information that would prevent successful execution.

CRITICAL EVALUATION CRITERIA:
1. **Parameter Completeness**: Does the goal provide all required parameters for each tool in the sequence?
2. **Information Clarity**: Is all necessary information clearly stated and unambiguous?
3. **Sequence Logic**: Does the goal naturally lead to the specified tool sequence?
4. **Context Sufficiency**: Is there enough context to understand what each tool should do?
5. **Ambiguity Resolution**: Are there any ambiguous terms or unclear references?
6. **Data Requirements**: Are all required data sources, files, or inputs specified?
7. **Output Expectations**: Are the expected outputs or results clearly defined?

RESPONSE FORMAT:
You must respond with a JSON object in this exact format:
{
    "missing_information": ["<list of specific missing information>"],
    "reasoning": "<detailed explanation of your assessment>"
}

IMPORTANT: Be thorough and identify ALL missing information. Each missing information item should be a specific, actionable detail that would prevent execution. Don't be lenient - if something is assumed but not explicitly stated, include it as missing."""

                user_prompt = f"""USER GOAL:
{user_goal}

ENVIRONMENT EXPECTATIONS:
{environment_expectations}

REQUIRED TOOL SEQUENCE:
{tool_info}

Please identify ALL missing information in the user goal and environment expectations that would prevent successful execution of this tool sequence."""

                response = await litellm.acompletion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **self.llm_config
                )
                
                response_text = response.choices[0].message.content.strip().replace('```json', '').replace('```', '')
                
                # Parse JSON response
                result_data = pyjson5.loads(response_text)
                
                missing_information = result_data.get("missing_information", [])
                reasoning = result_data.get("reasoning", "No reasoning provided")
                
                # Calculate score as number_of_issues / n_tools
                score = len(missing_information) / len(tool_sequence) if tool_sequence else 0.0
                
                return GoalUnambiguityResult(
                    score=score,
                    details={
                        "raw_response": response_text,
                        "n_tools": len(tool_sequence),
                        "n_missing_information": len(missing_information)
                    },
                    missing_information=missing_information,
                    reasoning=reasoning
                )
                    
            except Exception as e:
                logger.error(f"Error in goal unambiguity assessment: {e}")
                return GoalUnambiguityResult(
                    score=len(tool_sequence) if tool_sequence else 0.0,  # Assume all tools have issues if error
                    details={"error": str(e)},
                    missing_information=[],
                    reasoning="Assessment failed due to error",
                    error=str(e)
                )
    
    async def assess_batch(self, user_data: List[Dict[str, Any]]) -> List[GoalUnambiguityResult]:
        """
        Assess goal unambiguity for a batch of users.
        
        Args:
            user_data: List of user data dictionaries with 'user_goal', 'environment_expectations', and 'source'
            
        Returns:
            List of GoalUnambiguityResult objects
        """
        tasks = []
        for user in user_data:
            task = self.assess_unambiguity(
                user["user_goal"], 
                user["environment_expectations"],
                user["source"]
            )
            tasks.append(task)
        
        logger.info(f"Assessing goal unambiguity for {len(tasks)} users")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error assessing user {i}: {result}")
                processed_results.append(GoalUnambiguityResult(
                    score=len(user_data[i]["source"]) if user_data[i]["source"] else 0.0,  # Assume all tools have issues if error
                    details={"error": str(result)},
                    missing_information=[],
                    reasoning="Assessment failed due to exception",
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        logger.success(f"Completed goal unambiguity assessment for {len(processed_results)} users")
        return processed_results 