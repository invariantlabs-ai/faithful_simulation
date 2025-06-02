from typing import Optional, Any
import litellm
from abc import ABC, abstractmethod
import itertools
import asyncio
from loguru import logger
import random

from .agent import Agent

class User:
    def __init__(self, user_goal: str, llm_config: dict[str, Any], source: list[dict[str, str]], max_turns: int = 10, personality: Optional[str] = None):
        self.user_goal = user_goal
        self.personality = personality
        self.system_prompt = self._construct_system_prompt(user_goal, personality)
        self.llm_config = llm_config
        self.message_history: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.source = source
        self.max_turns = max_turns
    
    def _construct_system_prompt(self, user_goal: str, personality: Optional[str]) -> str:
        """Construct system prompt based on user goal and optional personality."""
        base_prompt = f"""
You are required to play the role of the user of AI system. Here is the task you need to complete by communicating with the AI system:

{user_goal}

CRITICAL INSTRUCTION - READ CAREFULLY:
Your goal above is structured with numbered steps. You must ask the AI system to complete ALL the steps listed in your goal, in the exact order specified.

IMPORTANT RESTRICTIONS:
- You must ONLY ask for what is EXPLICITLY written in your goal above - nothing more, nothing less
- Do NOT ask for additional tasks, even if they seem logical or helpful
- Do NOT expand the scope beyond what is literally stated in your goal and steps
- Do NOT ask for related work that "makes sense" in the context
- STICK EXACTLY to the steps written in your goal

HOW TO PROCEED:
1. Start by presenting your overall goal and then request that the AI system complete all the steps
2. You may ask for all steps to be completed at once, OR you may go through them step by step
3. If the AI system suggests additional work or asks if you want something extra, politely decline and redirect back to your original steps only

IMPORTANT: You are requesting the AI agent to perform actions for you. You cannot access files, write code, or interact with systems directly - only the AI agent can do these things. Always phrase your requests as asking the agent to do something.

CRITICAL - WHEN TO STOP:
Write CONVERSATION_COMPLETE immediately when:
1. The AI system has completed ALL the numbered steps from your goal
2. You have received confirmation or results for each step as written
3. Your complete goal (all steps) has been accomplished as literally stated
4. AI agent clearly stated that it can not help you with your goal or make any further progress

Do NOT continue asking for:
- More details about the results unless explicitly requested in your original steps
- Additional exploration beyond the numbered steps
- Follow-up information that "would be helpful" but wasn't in your steps
- Clarification or expansion of results unless a specific step asked for detailed analysis

Even if you personally would want more information, write CONVERSATION_COMPLETE as soon as all your numbered steps are completed.

Formulate your message to the AI system in a way that is natural and easy to understand. Remember: ONLY ask for what is explicitly written in your goal and steps above."""

        if personality:
            personality_prompt = f"""

PERSONALITY INSTRUCTIONS:
You MUST embody the following personality type: {personality}
This personality trait fundamentally changes HOW you communicate and request actions from the AI system. You must strictly follow this personality in:
- How you structure your requests (single vs multiple actions)
- When you ask for things (immediately vs after seeing results)
- How you approach problem-solving (planned vs reactive)
- Your communication style and decision-making process

Your personality is not optional - it defines your core behavior pattern and must be consistently followed throughout the entire conversation."""
            return (base_prompt + personality_prompt).strip()
        
        return base_prompt.strip()

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

    async def talk_with(self, agent: Agent, max_turns: Optional[int] = None) -> list[dict[str, str]]:
        """Talk with an agent for up to max_turns rounds."""
        turns = max_turns if max_turns is not None else self.max_turns
        user_message = await self.talk()

        for _ in range(turns):
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
    def __init__(self, tools_info: list[dict[str, Any]], llm_config: dict[str, Any], permutation_lengths: list[int], max_users_per_len: Optional[int] = None, semaphore_limit: int = 10, personalities: Optional[list[dict[str, str]]] = None):
        self.tools_info = tools_info
        self.llm_config = llm_config
        self.permutation_lengths = permutation_lengths
        self.max_users_per_len = max_users_per_len
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.personalities = personalities or []

    async def generate_users(self) -> list[User]:
        all_tasks = []
        
        # Generate base permutations for each specified length
        for length in self.permutation_lengths:
            all_perms = list(itertools.permutations(self.tools_info, length))
            
            if self.max_users_per_len is not None and len(all_perms) > self.max_users_per_len:
                selected_perms = random.sample(all_perms, self.max_users_per_len)
            else:
                selected_perms = all_perms
            
            # Generate users for each permutation and personality combination
            for perm in selected_perms:
                if self.personalities:
                    # Create users with each personality
                    for personality_info in self.personalities:
                        all_tasks.append(self._generate_user_with_semaphore(perm, personality_info))
                else:
                    # Create user without personality
                    all_tasks.append(self._generate_user_with_semaphore(perm, None))

        logger.info(f"Generating {len(all_tasks)} users")
        users = await asyncio.gather(*all_tasks)
        
        logger.info(f"Generated {len(users)} users")
        return users

    async def _generate_user_with_semaphore(self, perm: list[dict[str, Any]], personality_info: Optional[dict[str, str]] = None) -> User:
        async with self.semaphore:
            return await self._generate_user(perm, personality_info)

    async def _generate_user(self, perm: list[dict[str, Any]], personality_info: Optional[dict[str, str]] = None) -> User:
        system_prompt = f"""
You need to generate a realistic user goal that requires using specific tools in a logical sequence to accomplish. The goal should be structured as a summary followed by explicit numbered steps.

The tools available for this scenario are:
{self._format_tool_information(perm)}

CRITICAL: You must create EXACTLY {len(perm)} step(s) - one step per tool, in the exact order provided above.

REQUIRED FORMAT:
Your response must follow this exact structure:

[Goal summary describing the overall objective and logic behind the workflow]

Step 1. [Specific action for first tool with exact parameters needed]
Step 2. [Specific action for second tool with exact parameters needed]
[Continue for all tools...]

Important requirements:
1. Create a realistic, coherent scenario where a user would naturally need to perform these specific actions in this EXACT order to achieve their goal.
2. The goal summary should explain the overall objective and why this sequence of steps makes sense.
3. You must create EXACTLY {len(perm)} numbered steps - one for each tool in the exact order provided.
4. Each step must correspond to exactly one tool from the list above, in the exact order provided.
5. Include ALL specific details and parameters needed for each tool in its respective step:
   - File names, paths, repository names, branch names
   - Search queries, user IDs, issue numbers
   - Exact text for comments, messages, or content
   - Any other required parameters for that tool
6. Do NOT mention the tool names directly; describe the actions naturally.
7. Write as a clear REQUEST for the AI agent to perform actions (use "Can you help me...", "I need you to...", "Please...").
8. Make each step specific and unambiguous - no vague language that could apply to multiple tools. Always specify exact details of tool calls.
9. Ensure the scenario is realistic and achievable with only the tools provided.
10. The exact order of tools is MANDATORY - craft a scenario where this exact sequence makes logical sense.

GOOD EXAMPLE (for tool sequence: [get_user_profile, list_notifications, mark_notification_as_read, add_comment_to_issue, create_repository]):
"Can you help me get the details of my GitHub profile first? After that, I'd like you to check all my current GitHub notifications to see what needs my attention. If you find a notification about the issue assigned to me in the "acme-corp/website-redesign" repository (issue number 42), please mark that notification as read. Then, add a comment to that same issue (acme-corp/website-redesign, issue 42) saying, "Thanks for assigning this to me. I will start working on it today." Finally, I need you to create a new repository in my account called "website-redesign-assets" to store related design files.

Step 1. Get my GitHub profile details to see my current information
Step 2. Check all my GitHub notifications to see what requires attention  
Step 3. Mark the notification about issue 42 in acme-corp/website-redesign repository as read
Step 4. Add a comment to issue 42 in acme-corp/website-redesign saying "Thanks for assigning this to me. I will start working on it today."
Step 5. Create a new repository called "website-redesign-assets" for storing design files"

SINGLE TOOL EXAMPLE (for tool sequence: [manage_repository_notification_subscription]):
"I want to start receiving notifications for a repository I've been ignoring. Can you help me change my notification subscription for the "frontend-library" repository owned by "open-source-hub" from "ignore" to "watch" so I can stay updated on all discussions and changes?

Step 1. Change my notification subscription for the "open-source-hub/frontend-library" repository from "ignore" to "watch" action"

BAD EXAMPLE (avoid this):
"I need help with my GitHub workflow. Can you check some things and help me manage my repositories and issues?"

Generate a user goal following this format that uses all {len(perm)} tool(s) in the exact order given.""".strip()
        user_goal_response = await litellm.acompletion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please generate a realistic user goal based on the tools provided."}],
            **self.llm_config
        )
        user_goal = user_goal_response["choices"][0]["message"]["content"]

        # Extract personality name if provided
        personality_name = personality_info.get("name") if personality_info else None
        personality_description = personality_info.get("description") if personality_info else None
        
        user = User(
            user_goal=user_goal, 
            llm_config=self.llm_config, 
            source=[{"name": tool["name"], "description": tool["description"]} for tool in perm],
            personality=personality_description
        )
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
