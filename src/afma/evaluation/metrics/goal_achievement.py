from typing import Any, Dict, List, Optional
import litellm
from loguru import logger
import json

from .base import EvaluationResult


class GoalAchievementMetric:
    """
    LLM-based metric that assesses if the final environment state corresponds to the user's goal.
    
    This metric uses an LLM to evaluate whether the final state of the simulated environment
    successfully achieves what the user was trying to accomplish, regardless of the specific
    tools used to get there.
    """
    
    def __init__(self, 
                 llm_config: Dict[str, Any],
                 evaluation_prompt_template: Optional[str] = None):
        """
        Initialize the goal achievement metric.
        
        Args:
            llm_config: Configuration for litellm completion calls
            evaluation_prompt_template: Optional custom prompt template for evaluation
        """
        self.metric_name = "goal_achievement"
        self.llm_config = llm_config
        self.evaluation_prompt_template = evaluation_prompt_template or self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for goal achievement evaluation."""
        return """You are an expert evaluator assessing whether a user's goal has been successfully achieved based on the final state of a simulated environment.

TASK: Evaluate if the final environment state successfully accomplishes what the user was trying to achieve.

EVALUATION CRITERIA:
1. **Goal Achievement**: Does the final state reflect what the user wanted to accomplish?
2. **Completeness**: Is the goal fully achieved, or only partially?
3. **Relevance**: Are the changes in the environment state relevant to the user's goal?
4. **Quality**: Is the result of sufficient quality to satisfy the user's intent?

SCORING GUIDELINES:
- 1.0: Goal is completely and perfectly achieved
- 0.8-0.9: Goal is mostly achieved with minor issues
- 0.6-0.7: Goal is partially achieved with some significant gaps
- 0.4-0.5: Goal is minimally achieved with major issues
- 0.2-0.3: Goal is barely achieved or achieved incorrectly
- 0.0-0.1: Goal is not achieved at all

USER GOAL: {user_goal}

INITIAL ENVIRONMENT STATE: {environment_expectations}

FINAL ENVIRONMENT STATE:
{final_state}

Please provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your assessment>",
    "achievement_level": "<complete|mostly|partial|minimal|failed>",
    "key_factors": ["<factor1>", "<factor2>", ...],
    "suggestions": ["<suggestion1>", "<suggestion2>", ...]
}}

Focus on whether the final state successfully accomplishes the user's goal, comparing the initial and final environment states."""

    async def evaluate(self, 
                      user_goal: str, 
                      environment_state: List[Dict[str, Any]],
                      environment_expectations: str) -> EvaluationResult:
        """
        Evaluate if the final environment state achieves the user's goal.
        
        Args:
            user_goal: The original user goal/request
            environment_state: The final state of the simulated environment (from SimulatedEnvironment.state)
            environment_expectations: Initial environment state and expectations context
            
        Returns:
            EvaluationResult with score and details
        """
        try:
            # Format final environment state
            final_state = self._format_environment_state(environment_state)
            
            # Create the evaluation prompt
            prompt = self.evaluation_prompt_template.format(
                user_goal=user_goal,
                environment_expectations=environment_expectations,
                final_state=final_state
            )
            
            # Get LLM evaluation
            response = await litellm.acompletion(
                messages=[
                    {"role": "system", "content": "You are an expert evaluator assessing goal achievement in AI agent interactions."},
                    {"role": "user", "content": prompt}
                ],
                **self.llm_config
            )
            
            evaluation_text = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                evaluation_result = json.loads(evaluation_text)
                score = float(evaluation_result.get("score", 0.0))
                
                return EvaluationResult(
                    metric_name=self.metric_name,
                    score=score,
                    details={
                        "reasoning": evaluation_result.get("reasoning", ""),
                        "achievement_level": evaluation_result.get("achievement_level", "unknown"),
                        "key_factors": evaluation_result.get("key_factors", []),
                        "suggestions": evaluation_result.get("suggestions", []),
                        "raw_evaluation": evaluation_text,
                        "user_goal": user_goal,
                        "environment_state_summary": self._summarize_environment_state(environment_state)
                    }
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM evaluation response as JSON: {e}")
                # Fallback: try to extract a score from the text
                score = self._extract_score_from_text(evaluation_text)
                
                return EvaluationResult(
                    metric_name=self.metric_name,
                    score=score,
                    details={
                        "reasoning": evaluation_text,
                        "achievement_level": "unknown",
                        "key_factors": [],
                        "suggestions": [],
                        "raw_evaluation": evaluation_text,
                        "parse_error": str(e),
                        "user_goal": user_goal,
                        "environment_state_summary": self._summarize_environment_state(environment_state)
                    }
                )
            
        except Exception as e:
            logger.error(f"Error in goal achievement evaluation: {e}")
            return EvaluationResult(
                metric_name=self.metric_name,
                score=0.0,
                error=str(e)
            )
    
    def _format_environment_state(self, environment_state: List[Dict[str, Any]]) -> str:
        """Format environment state for the evaluation prompt."""
        if not environment_state:
            return "No environment state available"
        
        formatted_state = []
        for i, state_entry in enumerate(environment_state):
            if isinstance(state_entry, dict):
                if "tool_name" in state_entry:
                    formatted_state.append(f"Tool {i+1}: {state_entry['tool_name']}")
                    if "arguments" in state_entry:
                        formatted_state.append(f"  Arguments: {state_entry['arguments']}")
                    if "response" in state_entry:
                        formatted_state.append(f"  Result: {state_entry['response']}")
                else:
                    formatted_state.append(f"State Entry {i+1}: {json.dumps(state_entry, indent=2)}")
            else:
                formatted_state.append(f"State Entry {i+1}: {str(state_entry)}")
        
        return "\n".join(formatted_state) if formatted_state else "No environment state available"
    
    def _summarize_environment_state(self, environment_state: List[Dict[str, Any]]) -> str:
        """Create a summary of the environment state for details."""
        if not environment_state:
            return "No state changes"
        
        tool_names = []
        for entry in environment_state:
            if isinstance(entry, dict) and "tool_name" in entry:
                tool_names.append(entry["tool_name"])
        
        return f"Executed {len(environment_state)} operations: {', '.join(tool_names) if tool_names else 'No tools identified'}"
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract a score from text when JSON parsing fails."""
        import re
        
        # Look for patterns like "score: 0.8" or "0.8" or "80%"
        score_patterns = [
            r'score[:\s]*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)/1\.0',
            r'([0-9]+)%',
            r'([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if pattern == r'([0-9]+)%':
                        score = score / 100.0
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
        
        # Default fallback score
        return 0.5 