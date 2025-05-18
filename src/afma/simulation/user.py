from typing import Optional, Any
import litellm
from abc import ABC, abstractmethod
import itertools
import asyncio
from loguru import logger
import json

from .agent import Agent

class User:
    def __init__(self, user_goal: str, llm_config: dict[str, Any], source: list[dict[str, str]]):
        self.user_goal = user_goal
        self.system_prompt = f"""
You are required to play the role of the user of AI system. Here is the task you need to complete by communicating with the AI system:
{user_goal}

Formulate your message to the AI system, which I will emulate for you, in a way that is natural and easy to understand. Write CONVERSATION_COMPLETE as soon as you want to finish conversation either because task is done or because you understand that AI system is weird or won't be able to help you.
""".strip()
        self.llm_config = llm_config
        self.message_history: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.source = source
        logger.info(f"User initialized with task: {user_goal}")

    def __repr__(self) -> str:
        return f"User(user_goal={self.user_goal})"

    def __str__(self) -> str:
        return self.__repr__()

    async def talk(self, agent_message: Optional[str] = None) -> str:
        if agent_message:
            self.message_history.append({"role": "user", "content": agent_message})
        response = await litellm.acompletion(
            messages=self.message_history,
            **self.llm_config
        )
        user_message = response["choices"][0]["message"]["content"]
        self.message_history.append({"role": "assistant", "content": user_message})
        return user_message

    async def talk_with(self, agent: Agent, max_turns: int = 10) -> list[dict[str, str]]:
        user_message = await self.talk()

        for _ in range(max_turns):
            agent_response = await agent.talk(user_message)

            if "CONVERSATION_COMPLETE" in agent_response:
                break

            user_message = await self.talk(agent_response)

            if "CONVERSATION_COMPLETE" in user_message:
                break

        return agent.message_history

    def get_history(self) -> list[dict[str, str]]:
        return self.message_history


class UserSet(ABC):
    @abstractmethod
    async def generate_users(self) -> list[User]:
        pass


class CombinatoricUserSet(UserSet):
    def __init__(self, tools_info: list[dict[str, Any]], llm_config: dict[str, Any], max_permutation_length: int = 3):
        self.tools_info = tools_info
        self.llm_config = llm_config
        self.max_permutation_length = max_permutation_length

    async def generate_users(self) -> list[User]:
        tasks = []
        for length in range(1, self.max_permutation_length + 1):
            for perm in itertools.permutations(self.tools_info, length):
                tasks.append(self._generate_user(perm))
                break # TODO: Remove this
            break # TODO: Remove this

        users = await asyncio.gather(*tasks)

        return users

    def _format_tool_permutation(self, perm: list[dict[str, Any]]) -> str:
        formatted_string = "Execute the following tools in this exact order:\\n"
        for i, tool in enumerate(perm):
            name = tool["name"]
            description = tool["description"]
            formatted_string += f"{i+1}. {name}: {description}\\n"

            if "inputSchema" in tool and "required" in tool["inputSchema"] and tool["inputSchema"]["required"]:
                required_params = tool["inputSchema"]["required"]
                formatted_string += f"   Required parameters:\\n"
                properties = tool["inputSchema"]["properties"]
                for param_name in required_params:
                    param_description = properties[param_name].get("description", "No description available.")
                    formatted_string += f"     - {param_name}: {param_description}\\n"
        return formatted_string.strip()

    async def _generate_user(self, perm: list[dict[str, Any]]) -> User:
        system_prompt = f"""
You need to generate a specific, concrete task that logically leads to the use of these tools in the EXACT order specified in the user message. It is CRITICAL that the task strictly follows this sequence. Do NOT reverse the order or change it in any way.

{self._format_tool_permutation(perm)}

Important requirements:
1. Include specific, concrete examples (like exact file names, specific data points, etc.)
2. Do NOT mention specific tool names in your task, only use their descriptions to infer the actions.
3. Make your task detailed and specific enough that it would naturally lead to using these tools in this exact order.
4. Include placeholder information for tool parameters (e.g., specific file paths like "example.txt")
5. For file operations, always include a specific filename.
6. For web searches, specify exact search terms.
7. For any data operations, specify the exact data needed.

Example (for file reading): "Read file example.txt in my filesystem and send me a summary of its contents"
Example (for web search + file write): "Find information about climate change impacts on coral reefs and save the key points to a file named coral_research.txt"

Write the task in clear, natural language as a specific instruction.
""".strip()
        logger.info(f"Generating user with tools: {self._format_tool_permutation(perm)}")
        user_goal_response = await litellm.acompletion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please generate a user task based on the tools listed above."}],
            **self.llm_config
        )
        user_goal = user_goal_response["choices"][0]["message"]["content"]

        user = User(user_goal=user_goal, llm_config=self.llm_config, source=[{"name": tool["name"], "description": tool["description"]} for tool in perm])
        return user