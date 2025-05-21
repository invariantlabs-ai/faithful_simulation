from typing import Optional, Any
import litellm
from abc import ABC, abstractmethod
import itertools
import asyncio
from loguru import logger
import json
import random

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
    def __init__(self, tools_info: list[dict[str, Any]], llm_config: dict[str, Any], max_permutation_length: int = 3, max_users_per_len: Optional[int] = None, semaphore_limit: int = 10):
        self.tools_info = tools_info
        self.llm_config = llm_config
        self.max_permutation_length = max_permutation_length
        self.max_users_per_len = max_users_per_len
        self.semaphore = asyncio.Semaphore(semaphore_limit)

    async def generate_users(self) -> list[User]:
        all_tasks = []
        
        for length in range(1, self.max_permutation_length + 1):
            all_perms = list(itertools.permutations(self.tools_info, length))
            
            if self.max_users_per_len is not None and len(all_perms) > self.max_users_per_len:
                selected_perms = random.sample(all_perms, self.max_users_per_len)
            else:
                selected_perms = all_perms
            
            for perm in selected_perms:
                all_tasks.append(self._generate_user_with_semaphore(perm))

        logger.info(f"Generating {len(all_tasks)} users")
        users = await asyncio.gather(*all_tasks)

        return users

    async def _generate_user_with_semaphore(self, perm: list[dict[str, Any]]) -> User:
        async with self.semaphore:
            return await self._generate_user(perm)

    async def _generate_user(self, perm: list[dict[str, Any]]) -> User:
        system_prompt = f"""
You need to generate a realistic user goal that requires using specific tools in a logical sequence to accomplish. The goal should be coherent, natural, and represent a real user need rather than just a series of disconnected steps.

The tools available for this scenario are:
{self._format_tool_information(perm)}

Important requirements:
1. Create a realistic, coherent scenario where a user would naturally need to perform these specific actions in this EXACT order to achieve their goal.
2. Don't just list steps - create a narrative where each action logically leads to the next one.
3. Include specific details (file names, search terms, data points) that make the scenario concrete.
4. Do NOT mention the tool names directly; refer only to the actions a user would naturally want to perform.
5. Focus on the user's actual goal (what they want to accomplish) rather than the tools they need to use.
6. CRITICAL: Make sure the goal requires using ALL the specified tools in EXACTLY the order provided - the order is non-negotiable.
7. The goal should read as a user's request for help, not as instructions for using tools.
8. CRITICAL: For each tool listed above, make sure to include all context needed for its required parameters in the user's goal. For example, if edit_file requires specific edits, make sure the user mentions what changes they want to make.
9. CRITICAL: Ensure the user goal is realistic and achievable with only the tools provided. Don't create scenarios that would require additional tools not in the list.
10. The exact order of tools is MANDATORY - do not rearrange them for logical flow. Instead, craft a scenario where this exact sequence makes sense.

Example of good user goal:
"I'm working on a research paper about climate trends. I need to analyze the data in climate_data.csv, then create a visualization of the temperature changes over time, and finally export my findings as a PDF report that I can share with my colleagues."

Write the user goal in natural language as if a real person is describing what they want to achieve.
""".strip()
        user_goal_response = await litellm.acompletion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please generate a realistic user goal based on the tools provided."}],
            **self.llm_config
        )
        user_goal = user_goal_response["choices"][0]["message"]["content"]
        logger.debug(f"Prompt: {system_prompt}\nUser goal: {user_goal}")

        user = User(user_goal=user_goal, llm_config=self.llm_config, source=[{"name": tool["name"], "description": tool["description"]} for tool in perm])
        return user

    def _format_tool_information(self, perm: list[dict[str, Any]]) -> str:
        """Format tool information for generating coherent user goals."""
        formatted_string = ""
        for i, tool in enumerate(perm):
            name = tool["name"]
            description = tool["description"]
            formatted_string += f"{i+1}. {name}: {description}\n"

            if "inputSchema" in tool and "required" in tool["inputSchema"] and tool["inputSchema"]["required"]:
                required_params = tool["inputSchema"]["required"]
                formatted_string += f"   Function parameters:\n"
                properties = tool["inputSchema"]["properties"]
                for param_name in required_params:
                    param_description = properties[param_name].get("description", None)
                    if param_description is None:
                        formatted_string += f"     - {param_name}\n"
                    else:
                        formatted_string += f"     - {param_name}: {param_description}\n"
        return formatted_string.strip()
