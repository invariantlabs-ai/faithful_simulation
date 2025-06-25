from typing import Optional, Any
import litellm
from abc import ABC, abstractmethod
import itertools
import asyncio
from loguru import logger
import random

from .agent import Agent

class User:
    def __init__(self, user_goal: str, environment_expectations: str, llm_config: dict[str, Any], source: list[dict[str, str]], max_turns: int = 10, personality_name: Optional[str] = None, personality: Optional[str] = None):
        self.user_goal = user_goal
        self.environment_expectations = environment_expectations
        self.personality_name = personality_name
        self.personality = personality
        self.system_prompt = self._construct_system_prompt(user_goal, environment_expectations, personality_name, personality)
        self.llm_config = llm_config
        self.message_history: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.source = source
        self.max_turns = max_turns
    
    def _construct_system_prompt(self, user_goal: str, environment_expectations: str, personality_name: Optional[str], personality: Optional[str]) -> str:
        """Construct system prompt based on user goal, environment expectations, and optional personality."""
        base_prompt = f"""
You are required to play the role of the user of AI system. Here is the task you need to complete by communicating with the AI system:

{user_goal}

ENVIRONMENT EXPECTATIONS:
{environment_expectations}

CRITICAL INSTRUCTION - READ CAREFULLY:
Your goal above describes a complete workflow that you want the AI system to accomplish. You must ask the AI system to complete the entire workflow described in your goal.

IMPORTANT RESTRICTIONS:
- You must ONLY ask for what is EXPLICITLY written in your goal above - nothing more, nothing less
- Do NOT ask for additional tasks, even if they seem logical or helpful
- Do NOT expand the scope beyond what is literally stated in your goal
- Do NOT ask for related work that "makes sense" in the context
- Do NOT add extra details, clarifications, or requests beyond what is literally written in your goal
- STICK EXACTLY to what is written in your goal without any embellishment or additional requests

HOW TO PROCEED:
1. Start by presenting your overall goal and request that the AI system complete the entire workflow
2. You may ask for the complete workflow to be done at once, OR you may break it down into logical parts if the goal is complex
3. When making requests, use ONLY the exact wording and scope from your goal - do not add extra requests for status, details, or related information
4. If the AI system suggests additional work or asks if you want something extra, politely decline and redirect back to your original goal only
5. Complete each logical part of your goal fully before moving to the next part

CRITICAL ERROR HANDLING INSTRUCTIONS:
When the AI agent encounters errors, failures, or issues (like rate limits, server errors, or tool failures):
- DO NOT be passive or polite about waiting
- DO NOT say things like "I'll wait for you" or "let me know when you're ready"
- DO NOT offer to help with other things or suggest waiting
- INSTEAD, immediately push the agent to retry and complete the goal
- Use direct, assertive language like:
  * "Please retry that action and continue with the task"
  * "Try again and complete what I asked for"
  * "Retry the operation and finish the job"
  * "Please try again and get this done"
- Your goal is NOT accomplished until the AI agent successfully completes ALL actions described in your goal
- Keep pushing until the goal is actually achieved or the agent clearly states it cannot help

IMPORTANT: You are simulating a user and are requesting the AI agent to perform actions for you. You cannot access files, write code, or interact with systems directly - only the AI agent can do these things. Always phrase your requests as asking the agent to do something.

Write CONVERSATION_COMPLETE immediately when:
1. The AI system has completed ALL the actions described in your goal
2. You have received confirmation or results for the complete workflow
3. Your complete goal has been accomplished as literally stated
4. AI agent clearly stated that it can not help you with your goal or make any further progress

Do NOT continue asking for:
- More details about the results unless explicitly requested in your original goal
- Additional exploration beyond what was described in your goal
- Follow-up information that "would be helpful" but wasn't in your goal
- Clarification or expansion of results unless your goal specifically asked for detailed analysis

Even if you personally would want more information, write CONVERSATION_COMPLETE as soon as your complete goal has been accomplished.

Formulate your message to the AI system in a way that is natural and easy to understand. Remember: ONLY ask for what is explicitly written in your goal above."""

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
    def __init__(self, tools_info: list[dict[str, Any]], generation_llm_config: dict[str, Any], simulation_llm_config: dict[str, Any], permutation_lengths: list[int], max_users_per_len: Optional[int] = None, semaphore_limit: int = 10, personalities: Optional[list[dict[str, str]]] = None, random_seed: Optional[int] = 42):
        self.tools_info = tools_info
        self.generation_llm_config = generation_llm_config
        self.simulation_llm_config = simulation_llm_config
        self.permutation_lengths = permutation_lengths
        self.max_users_per_len = max_users_per_len
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.personalities = personalities or []
        self._rng = random.Random(random_seed) if random_seed is not None else random
        self._tool_graph = None  # Cache for the tool relationship graph

    async def generate_users(self) -> list[User]:
        all_tasks = []
        
        # Generate the tool relationship graph once
        if self._tool_graph is None:
            self._tool_graph = await self._generate_tool_graph()
        
        # Generate reasonable tool sequences for each specified length
        for length in self.permutation_lengths:
            if self.max_users_per_len is not None:
                num_sequences = min(self.max_users_per_len, len(self.tools_info) ** length)
            else:
                num_sequences = len(self.tools_info) ** length
            
            # Generate multiple reasonable sequences for this length
            for _ in range(num_sequences):
                if self.personalities:
                    # Create users with each personality
                    for personality_info in self.personalities:
                        all_tasks.append(self._generate_user_with_semaphore(length, personality_info))
                else:
                    # Create user without personality
                    all_tasks.append(self._generate_user_with_semaphore(length, None))

        logger.info(f"Generating {len(all_tasks)} users")
        users = await asyncio.gather(*all_tasks)
        
        logger.info(f"Generated {len(users)} users")
        return users

    async def _generate_user_with_semaphore(self, sequence_length: int, personality_info: Optional[dict[str, str]] = None) -> User:
        async with self.semaphore:
            return await self._generate_user(sequence_length, personality_info)

    async def _generate_tool_graph(self) -> dict[str, list[str]]:
        """Generate a directed graph of tool relationships using a single LLM call."""
        tools_info = sorted(self.tools_info, key=lambda _: self._rng.random()) # Shuffle the tools to avoid bias
        system_prompt = f"""
You are an expert at understanding tool workflows and determining logical sequences of actions. Given a set of tools, you need to create a directed graph showing which tools logically follow other tools in realistic workflows.

AVAILABLE TOOLS:
{self._format_tool_information(tools_info, include_parameters=False)}

TASK: Create a directed graph showing which tools logically follow other tools in realistic user workflows.

CRITICAL GUIDELINES:
1. **Focus on user intent, not tool similarity**: Connect tools based on what users actually want to accomplish, not because tools are similar
2. **Prioritize cross-domain workflows**: Users often move between different types of tools (search → write, get info → take action, etc.)
3. **Avoid over-connecting similar tools**: Don't connect every tool to every other tool in the same category
4. **Think in terms of realistic user goals**: What would a user actually do next to achieve their objective?
5. **Limit connections per tool**: Most tools should have 3-5 outgoing connections, not 10+

KEY WORKFLOW PATTERNS TO PRIORITIZE:
- **Information Gathering → Action**: Search → Write/Edit/Create
- **Setup → Work**: Initialize → Perform operations
- **Read → Modify**: Get content → Change/Update content
- **Create → Manage**: Create resources → Manage/Update them
- **Check → Act**: Verify status → Take action based on results

EXAMPLES OF REALISTIC CROSS-DOMAIN CONNECTIONS:
- Search tools → File writing/editing tools (research and implement)
- User profile tools → Repository creation tools (setup and start working)
- File reading tools → File editing tools (examine and modify)
- Repository tools → File management tools (create repo and add files)
- Search tools → File writing/editing tools (find information and process it)
- GitHub file reading → Filesystem writing (download and save locally)
- Filesystem writing → GitHub file operations (local changes → push to repo)
- Filesystem editing → GitHub file operations (edit locally → commit changes)

SPECIFIC WORKFLOW PATTERNS TO INCLUDE:
- **Download → Work**: Get file from GitHub → Write/Edit locally
- **Local → Remote**: Write/Edit files locally → Push to GitHub
- **Setup → Verify**: Create directory → List contents to verify
- **Push → Check**: Push changes → List commits to verify
- **Profile → Action**: Get user info → Create repositories or other actions
- **Read → Act**: Read file info → Edit or modify the file

GENERAL WORKFLOW PATTERNS (apply to any AI agent):
- **Create → Verify**: After creating resources, verify they exist/work
- **Search → Multiple Actions**: Search results should lead to diverse actions, not just one type
- **Setup → Multiple Work Options**: Initialization should enable various work activities
- **Read → Multiple Actions**: Reading should enable various follow-up actions
- **Action → Status Check**: After taking actions, check the status/results
- **Discovery → Implementation**: Find information → Implement based on findings

AVOID THESE PATTERNS:
- Don't create dense clusters within tool categories
- Don't connect tools just because they're similar

RESPONSE FORMAT:
Respond with ONLY the directed graph in this exact format:
tool_name_1: [next_tool_name_1, next_tool_name_2, ...]
tool_name_2: [next_tool_name_3, next_tool_name_4, ...]
...

Rules:
- Use EXACT tool names as shown in the AVAILABLE TOOLS section
- Include ALL tools in the graph, even if they have no outgoing edges (empty list)
- Do not include explanations or additional text
- Each line should follow the exact format: "tool_name: [list, of, next, tools]"
- Focus on realistic user workflows, not tool similarity
- Limit to 2-5 outgoing connections per tool unless there's a strong workflow reason for more""".strip()

        try:
            response = await litellm.acompletion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please generate the directed graph of tool relationships."}
                ],
                **self.generation_llm_config
            )
            
            response_text = response["choices"][0]["message"]["content"].strip()
            
            # Parse the response to extract the graph
            graph = {}
            for line in response_text.split('\n'):
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        tool_name = parts[0].strip()
                        next_tools_str = parts[1].strip()
                        
                        # Parse the list of next tools
                        if next_tools_str.startswith('[') and next_tools_str.endswith(']'):
                            next_tools_str = next_tools_str[1:-1]  # Remove brackets
                            if next_tools_str.strip():
                                next_tools = [tool.strip() for tool in next_tools_str.split(',')]
                            else:
                                next_tools = []
                        else:
                            next_tools = []
                        
                        graph[tool_name] = next_tools
            
            # Validate that all tools are included in the graph
            tool_names = {tool["name"] for tool in self.tools_info}
            for tool_name in tool_names:
                if tool_name not in graph:
                    logger.warning(f"Tool {tool_name} not found in graph")
                    graph[tool_name] = []
            
            logger.info(f"Generated tool graph with {len(graph)} nodes")
            return graph
            
        except Exception as e:
            logger.warning(f"Error generating tool graph: {e}. Falling back to random graph.")
            # Fallback: create a simple random graph
            return self._create_fallback_graph()

    def _create_fallback_graph(self) -> dict[str, list[str]]:
        """Create a fallback graph when LLM generation fails."""
        tool_names = [tool["name"] for tool in self.tools_info]
        graph = {}
        
        for tool_name in tool_names:
            # Randomly select 0-3 tools to follow this one
            num_next = self._rng.randint(0, min(3, len(tool_names) - 1))
            if num_next > 0:
                next_tools = self._rng.sample([t for t in tool_names if t != tool_name], num_next)
            else:
                next_tools = []
            graph[tool_name] = next_tools
        
        return graph

    def _generate_reasonable_tool_sequence(self, length: int) -> list[dict[str, Any]]:
        """Generate a reasonable tool sequence using the pre-computed tool graph."""
        sequence = []
        available_tools = {tool["name"]: tool for tool in self.tools_info}
        
        # Step 1: Sample first tool randomly
        first_tool_name = self._rng.choice(list(available_tools.keys()))
        sequence.append(available_tools[first_tool_name])
        del available_tools[first_tool_name]
        
        # Steps 2-3: For each subsequent tool, use the graph to find reasonable next tools, then sample randomly
        for i in range(1, length):
            if not available_tools:
                break
            
            # Get the last tool in the sequence
            last_tool_name = sequence[-1]["name"]
            
            # Get reasonable next tools from the graph
            if last_tool_name in self._tool_graph:
                reasonable_tool_names = self._tool_graph[last_tool_name]
                # Filter to only include tools that are still available
                available_reasonable_tools = [name for name in reasonable_tool_names if name in available_tools]
            else:
                available_reasonable_tools = []
            
            if not available_reasonable_tools:
                # If no reasonable tools in graph or none available, fall back to random selection
                available_reasonable_tools = list(available_tools.keys())
            
            # Sample randomly from the reasonable tools
            next_tool_name = self._rng.choice(available_reasonable_tools)
            sequence.append(available_tools[next_tool_name])
            del available_tools[next_tool_name]
        
        return sequence

    async def _generate_user(self, sequence_length: int, personality_info: Optional[dict[str, str]] = None) -> User:
        # Generate a reasonable tool sequence
        perm = self._generate_reasonable_tool_sequence(sequence_length)
        
        # Get tools not in the current permutation
        perm_tool_names = {tool["name"] for tool in perm}
        other_tools = [tool for tool in self.tools_info if tool["name"] not in perm_tool_names]
        
        system_prompt = f"""
You need to generate a realistic user goal that requires using specific tools in a logical sequence to accomplish. The goal should be written in a natural, conversational style - like how a real user would actually ask for help.

The tools REQUIRED for this scenario are (in exact order):
{self._format_tool_information(perm, include_parameters=True)}

CRITICAL: You must create a scenario that naturally requires ALL {len(perm)} tool(s) in the exact order provided above.

IMPORTANT CONTEXT - Other available tools in the system that you must AVOID using:
{self._format_tool_information(other_tools, include_parameters=False)}

CRITICAL INSTRUCTION: When crafting your user goal, you MUST ensure that the scenario specifically requires the EXACT sequence of required tools listed above and would NOT be better solved using any of the other available tools. Do NOT create scenarios where any of the other available tools would be more optimal, logical, or natural to use. 

For example, if there's a specialized web crawling tool in the "other available tools" section, do NOT create a web crawling scenario - instead create a scenario where the required tools are the best fit. The goal should be crafted so that using the required tools in the specified order is the ONLY logical and effective approach, and using any other available tool would be suboptimal or inappropriate.

CRITICAL USER ROLE RESTRICTION:
- The user CANNOT perform any actions themselves (no file editing, no manual work, no local operations)
- The user can ONLY guide the AI agent to do all the work
- NEVER include phrases like "so I can edit it locally", "after I make changes", "when I finish", etc.
- The user must provide all necessary information upfront and ask the agent to handle everything
- All actions must be performed by the AI agent, not the user

IMPORTANT: The user already has all necessary information and context. Do NOT create scenarios that would require additional tools or information gathering steps that are not in the provided list. The user should provide all required details directly in their goal.

REQUIRED FORMAT:
Your response must be a single, natural paragraph that describes what the user wants to accomplish, followed by environment expectations. It should:

1. Start with a clear, conversational description of the overall goal
2. Include all necessary context and assumptions about the environment
3. Provide all required parameters for the tools in a natural way
4. Flow logically from one action to the next without explicit step numbering
5. Sound like how a real user would actually ask for help
6. ALWAYS include an "ENVIRONMENT_EXPECTATIONS:" section at the end listing key assumptions

Important requirements:
1. Create a realistic, coherent scenario where a user would naturally need to perform these specific actions in this EXACT order to achieve their goal.
2. Write in a conversational, natural tone - avoid robotic or overly formal language.
3. You must create a scenario that requires ALL {len(perm)} tool(s) in the exact order provided.
4. Each tool must be naturally integrated into the workflow - don't force them artificially.
5. CRITICAL: Include EXACTLY the parameters listed in each tool's "Function parameters" section - no more, no less.
6. Look at each tool's required and optional parameters carefully and provide only those specific details.
7. Do NOT mention the tool names directly; describe the actions naturally.
8. Write as a clear REQUEST for the AI agent to perform actions (use "Can you help me...", "I need you to...", "Please...").
9. Make the scenario specific and unambiguous - no vague language that could apply to multiple tools.
10. Ensure the scenario is realistic and achievable with only the required tools provided.
11. The exact order of tools is MANDATORY - craft a scenario where this exact sequence makes logical sense.
12. CRITICAL: Do NOT assume the agent needs to gather information first. The user should already know all necessary details and provide them in the goal.
13. CRITICAL: The scenario must be designed so that using the required tools in the specified order is MORE APPROPRIATE than using any of the other available tools. Avoid creating scenarios that would be better solved with alternative tools from the system.
14. ALWAYS include an "ENVIRONMENT_EXPECTATIONS:" section at the end with numbered assumptions about what exists or is available.
15. CRITICAL: The user must provide all necessary content, data, or information upfront - never assume the user will provide anything later or do any work themselves.

GOOD EXAMPLE (for tool sequence: [list_branches, create_branch, push_files]):
"User wants to edit a branch "better_calendar_ui" done by a senior colleague in a repository "calendar_builders/calendar_app". Can the AI first check that this branch exists in the repo? Then the user needs the AI to create a new branch "better_calendar_ui_new_components" from that branch. Finally, the user wants the AI to push [{{path: "src/ui/component.js", content: "export function Component() {{ return <div>New UI</div>; }}"}}, {{path: "src/ui/styles.css", content: ".new-ui {{ color: blue; }}"}}] files to this new branch with a commit message "Add new UI component and styles". 

ENVIRONMENT_EXPECTATIONS:
1. Repository "calendar_builders/calendar_app" exists and user has access to that
2. There exists branch "better_calendar_ui" in this repo"

GOOD EXAMPLE (for tool sequence: [get_user_profile, list_notifications, mark_notification_as_read, add_comment_to_issue, create_repository]):
"User needs to check their GitHub profile and then handle some notifications. Can the AI get the user's profile details first, then check all their notifications? If the AI finds a notification about issue 42 in the "acme-corp/website-redesign" repository, please mark that notification as read and add a comment to that issue saying "Thanks for assigning this to me. I will start working on it today." Finally, the user needs the AI to create a new repository called "website-redesign-assets" for storing design files.

ENVIRONMENT_EXPECTATIONS:
1. User has a GitHub account with profile information
2. Repository "acme-corp/website-redesign" exists and user has access to it
3. Issue 42 exists in the repository"

SINGLE TOOL EXAMPLE (for tool sequence: [manage_repository_notification_subscription]):
"User wants to start receiving notifications for a repository they've been ignoring. Can the AI help the user change their notification subscription for the "frontend-library" repository owned by "open-source-hub" from "ignore" to "watch" so the user can stay updated on all discussions and changes?

ENVIRONMENT_EXPECTATIONS:
1. Repository "open-source-hub/frontend-library" exists and user has access to it
2. User currently has notification subscription set to "ignore" for this repository"

Generate a user goal following this format that uses all {len(perm)} tool(s) in the exact order given.""".strip()
        user_goal_response = await litellm.acompletion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please generate a realistic user goal based on the tools provided."}],
            **self.generation_llm_config
        )
        full_user_goal = user_goal_response["choices"][0]["message"]["content"]

        # Parse the user goal to extract goal and environment expectations
        goal_part, environment_expectations = self._parse_user_goal(full_user_goal)

        # Extract personality name if provided
        personality_name = personality_info.get("name") if personality_info else None
        personality_description = personality_info.get("description") if personality_info else None
        
        user = User(
            user_goal=goal_part, 
            environment_expectations=environment_expectations,
            llm_config=self.simulation_llm_config, 
            source=[{"name": tool["name"], "description": tool["description"]} for tool in perm],
            personality_name=personality_name,
            personality=personality_description
        )
        return user

    def _parse_user_goal(self, full_user_goal: str) -> tuple[str, str]:
        """Parse the full user goal into goal part and environment expectations part."""
        # Split by "ENVIRONMENT_EXPECTATIONS:" to separate the parts
        if "ENVIRONMENT_EXPECTATIONS:" in full_user_goal:
            parts = full_user_goal.split("ENVIRONMENT_EXPECTATIONS:", 1)
            goal_part = parts[0].strip()
            environment_expectations = parts[1].strip()
        else:
            # If no environment expectations found, use the full text as goal and empty expectations
            goal_part = full_user_goal.strip()
            logger.warning(f"No environment expectations found in user goal: {full_user_goal}")
            environment_expectations = "No specific environment expectations provided"
        
        return goal_part, environment_expectations

    def _format_tool_information(self, perm: list[dict[str, Any]], include_parameters: bool = True) -> str:
        """Format tool information for generating coherent user goals."""
        formatted_string = ""
        for i, tool in enumerate(perm):
            name = tool["name"]
            description = tool["description"]
            formatted_string += f"{i+1}. {name}: {description}\n"

            if include_parameters and "inputSchema" in tool and "properties" in tool["inputSchema"]:
                properties = tool["inputSchema"]["properties"]
                required_params = tool["inputSchema"].get("required", [])
                
                if properties:
                    formatted_string += f"   Function parameters:\n"
                    for param_name, param_info in properties.items():
                        param_description = param_info.get("description", "")
                        is_required = param_name in required_params
                        required_label = " (required)" if is_required else " (optional)"
                        
                        if param_description:
                            formatted_string += f"     - {param_name}{required_label}: {param_description}\n"
                        else:
                            formatted_string += f"     - {param_name}{required_label}\n"
        return formatted_string.strip()
