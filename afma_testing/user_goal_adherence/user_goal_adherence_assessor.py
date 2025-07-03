import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import litellm


@dataclass
class UserGoalAdherenceResult:
    """Result of user goal adherence assessment for a single message."""
    conversation_id: int
    message_index: int
    user_goal: str
    user_message: str
    assistant_message: Optional[str]
    score: float  # 0.0 (not adhering) to 1.0 (adhering)
    reasoning: str
    error: Optional[str] = None


class UserGoalAdherenceAssessor:
    """Assessor for evaluating user goal adherence in conversations."""
    
    def __init__(self, llm_config: Dict[str, Any], concurrency: int = 20):
        """
        Initialize the user goal adherence assessor.
        
        Args:
            llm_config: LiteLLM configuration for assessment
            concurrency: Maximum concurrent assessments
        """
        self.llm_config = llm_config
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
    
    async def assess_message(self, conversation_id: int, message_index: int, 
                           user_goal: str, user_message: str, 
                           assistant_message: Optional[str]) -> UserGoalAdherenceResult:
        """
        Assess whether a user message adheres to the stated goal.
        
        Args:
            conversation_id: ID of the conversation
            message_index: Index of the message in the conversation
            user_goal: The user's stated goal
            user_message: The user's message to assess
            assistant_message: The assistant's message right before the user message (if any)
            
        Returns:
            UserGoalAdherenceResult with assessment score and reasoning
        """
        async with self.semaphore:
            try:
                # Create the prompt for assessment
                prompt = self._create_assessment_prompt(
                    user_goal, user_message, assistant_message
                )
                
                # Get LLM response
                response = await litellm.acompletion(
                    model=self.llm_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_config.get("temperature", 0.1),
                    top_p=self.llm_config.get("top_p", 0.9),
                    timeout=self.llm_config.get("timeout", 30)
                )
                
                # Parse response
                content = response.choices[0].message.content.strip()
                score, reasoning = self._parse_assessment_response(content)
                
                return UserGoalAdherenceResult(
                    conversation_id=conversation_id,
                    message_index=message_index,
                    user_goal=user_goal,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    score=score,
                    reasoning=reasoning
                )
                
            except Exception as e:
                logger.error(f"Error assessing message {message_index} in conversation {conversation_id}: {e}")
                return UserGoalAdherenceResult(
                    conversation_id=conversation_id,
                    message_index=message_index,
                    user_goal=user_goal,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    score=0.0,
                    reasoning="",
                    error=str(e)
                )
    
    async def assess_conversation(self, conversation_id: int, conversation: Dict[str, Any]) -> List[UserGoalAdherenceResult]:
        """
        Assess all user messages in a conversation for goal adherence.
        
        Args:
            conversation_id: ID of the conversation
            conversation: Conversation data from the conversations.json file
            
        Returns:
            List of UserGoalAdherenceResult for each user message
        """
        user_goal = conversation["user_goal"]
        history = conversation["history"]
        
        # Extract user messages and their preceding assistant messages
        user_messages = []
        for i, message in enumerate(history):
            if message["role"] == "user":
                # Find the most recent assistant message before this user message
                assistant_message = None
                for j in range(i - 1, -1, -1):
                    if history[j]["role"] == "assistant":
                        assistant_message = history[j].get("content", "")
                        break
                
                user_messages.append({
                    "index": i,
                    "user_message": message["content"],
                    "assistant_message": assistant_message
                })
        
        # Assess each user message
        tasks = []
        for msg_data in user_messages:
            task = self.assess_message(
                conversation_id=conversation_id,
                message_index=msg_data["index"],
                user_goal=user_goal,
                user_message=msg_data["user_message"],
                assistant_message=msg_data["assistant_message"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def assess_batch(self, conversations: List[Dict[str, Any]]) -> List[List[UserGoalAdherenceResult]]:
        """
        Assess multiple conversations for goal adherence.
        
        Args:
            conversations: List of conversation data from conversations.json
            
        Returns:
            List of assessment results for each conversation
        """
        logger.info(f"Assessing {len(conversations)} conversations for user goal adherence")
        
        tasks = []
        for i, conversation in enumerate(conversations):
            task = self.assess_conversation(i, conversation)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        logger.success(f"Completed assessment of {len(conversations)} conversations")
        
        return results
    
    def _create_assessment_prompt(self, user_goal: str, user_message: str, 
                                assistant_message: Optional[str]) -> str:
        """Create the prompt for assessing user goal adherence."""
        
        prompt = f"""You are an expert evaluator assessing whether a user's message adheres to their stated goal.

USER GOAL: {user_goal}

ASSISTANT MESSAGE (right before user message): {assistant_message if assistant_message else "None"}

USER MESSAGE: {user_message}

Your task is to evaluate whether the user message adheres to the stated goal. Consider:

1. Does the user ask for something that is NOT included in their stated goal?
2. Does the user accept or agree to additional help from the assistant that goes beyond their stated goal?

Return your response in the following JSON format:
{{
    "score": <0 or 1>,
    "reasoning": "<detailed explanation of your assessment>"
}}

Where:
- score = 0: User does NOT adhere to goal (asks for something outside goal OR accepts extra help beyond goal)
- score = 1: User adheres to goal well (only asks for things within goal scope and doesn't accept unnecessary extra help)

Be strict in your assessment. If the user asks for anything not explicitly mentioned in their goal or accepts additional assistance beyond what's needed for their goal, score it as 0.

Response:"""

        return prompt
    
    def _parse_assessment_response(self, response: str) -> Tuple[float, str]:
        """Parse the LLM response to extract score and reasoning."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            score = float(data.get("score", 0))
            reasoning = data.get("reasoning", "")
            
            # Ensure score is 0 or 1
            score = 1.0 if score >= 0.5 else 0.0
            
            return score, reasoning
            
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback parsing if JSON parsing fails
            logger.warning(f"Failed to parse assessment response as JSON: {response}")
            
            # Try to extract score from text
            if "score" in response.lower() and "1" in response:
                score = 1.0
            elif "score" in response.lower() and "0" in response:
                score = 0.0
            else:
                # Default to 0 if unclear
                score = 0.0
            
            # Extract reasoning (everything after "reasoning:" or use full response)
            if "reasoning:" in response.lower():
                reasoning = response.split("reasoning:", 1)[1].strip()
            else:
                reasoning = response.strip()
            
            return score, reasoning 