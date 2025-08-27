import asyncio
import json
from typing import Any, Dict, List, Tuple

import litellm
from loguru import logger


class PersonalityAdherenceAssessor:
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.user_personality_assessment_prompt = """
You are a strict compliance auditor. Your task is to evaluate whether a user's messages, as documented in a conversation log, strictly follow a given personality profile.

Read the "User Personality" description carefully. This is the **required** behavior for the user.
Then, analyze the "User Messages" log.
Your goal is to determine how closely the user's messages match the described personality.

- A score of **1.0** means the user's messages perfectly and consistently followed ALL aspects of the personality description.
- A score of **0.0** means the user's messages showed NO signs of the described personality, or they did the complete opposite.
- Scores in between represent partial adherence.

For example, if the user personality is "impatient" and the user's messages are calm and patient, the adherence score should be very low (close to 0.0).

Please provide only the score as a single float number in your response.

**User Personality:**
{personality}

**User Messages:**
{messages}
"""
        self.environment_personality_assessment_prompt = """
You are a strict compliance auditor. Your task is to evaluate whether an environment's behavior, as documented in a series of tool interactions, strictly follows a given personality profile.

Read the "Environment Personality" description carefully. This is the **required** behavior.
Then, analyze the "Tool Interactions" log.
Your goal is to determine how closely the tool results match the described personality.

- A score of **1.0** means the environment perfectly and consistently followed ALL aspects of the personality description.
- A score of **0.0** means the environment's behavior showed NO signs of the described personality, or it did the complete opposite.
- Scores in between represent partial adherence.

For example, if the personality is "Adversarial" and is supposed to return deceptive text, but the tool results are all normal and helpful, the adherence score should be very low (close to 0.0). Conversely, if the personality is "Perfect" and the tool results are clean and correct, the adherence score should be high (close to 1.0).

Please provide only the score as a single float number in your response.

**Environment Personality:**
{personality}

**Tool Interactions:**
{interactions}
"""

    async def assess_conversation(self, conversation_trace: Dict[str, Any]) -> Tuple[float, float, str, str]:
        user_personality = conversation_trace.get("user_personality", "No user personality defined.")
        env_personality = conversation_trace.get("environment_personality", "No environment personality defined.")
        history = conversation_trace.get("history", [])

        user_messages = self._extract_user_messages(history)
        tool_interactions = self._extract_tool_interactions(history)

        user_score_task = self.assess_user_personality(user_personality, user_messages)
        env_score_task = self.assess_environment_personality(env_personality, tool_interactions)

        user_score, env_score = await asyncio.gather(user_score_task, env_score_task)

        return user_score, env_score, user_messages, tool_interactions

    def _extract_user_messages(self, history: List[Dict[str, Any]]) -> str:
        messages = [msg["content"] for msg in history if msg["role"] == "user" and msg["content"]]
        return "\n".join(messages)

    def _extract_tool_interactions(self, history: List[Dict[str, Any]]) -> str:
        interactions = []
        for msg in history:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    interactions.append(f"ASSISTANT TOOL CALL: {json.dumps(tool_call)}")
            elif msg["role"] == "tool":
                interactions.append(f"TOOL RESULT: {json.dumps(msg)}")
        return "\n".join(interactions)

    async def assess_user_personality(self, personality: str, messages: str) -> float: 
        prompt = self.user_personality_assessment_prompt.format(personality=personality, messages=messages)
        try:
            response = await litellm.acompletion(
                messages=[{"role": "user", "content": prompt}], **self.llm_config
            )
            score_str = response.choices[0].message.content.strip()
            return float(score_str)
        except Exception as e:
            logger.error(f"Error assessing user personality: {e}")
            return 0.0

    async def assess_environment_personality(self, personality: str, interactions: str) -> float:
        prompt = self.environment_personality_assessment_prompt.format(personality=personality, interactions=interactions)
        try:
            response = await litellm.acompletion(
                messages=[{"role": "user", "content": prompt}], **self.llm_config
            )
            score_str = response.choices[0].message.content.strip()
            return float(score_str)
        except Exception as e:
            logger.error(f"Error assessing environment personality: {e}")
            return 0.0 