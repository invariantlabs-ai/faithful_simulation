# Agent Failure Mode Analysis through User Simulation

## Overview
AI agents are becoming increasingly capable, yet remain error-prone and challenging to trust in real-world high-stakes applications. They often misinterpret user intent, behave unpredictably in complex workflows, and fail under ambiguous task specifications. Without a structured understanding of these failure modes, production deployment remains risky and ad hoc. In this project, we aim to provide this structured understanding via a systematic testing and simulation approach. Knowing the tools available to an agent, we can simulate both user input and tool output that explores error-prone and unsafe edge-cases.

## Two-pager
Testing AI agents in complex, high-stakes scenarios is a major challenge. Current methods are often insufficient. Manual testing by human users is slow, expensive, and difficult to scale, while static analysis of an agent's tools can reveal potential vulnerabilities but fails to capture the emergent, unpredictable errors that arise from the agent's own behavior and its interaction with a dynamic environment.

This project introduces a third approach: **agent failure mode analysis through automated user simulation**. We propose a framework that systematically tests an agent by simulating conversations between the agent and a variety of simulated users within a simulated environment. This allows for rapid, scalable, and reproducible testing that can uncover failure modes missed by other methods.

The framework is designed to be agent-agnostic. Given only the definition of an agent's toolset (via an MCP config file), it orchestrates an end-to-end evaluation pipeline. It begins by automatically generating realistic user goals and initial environment states. It then simulates the entire interaction—a user pursuing their goal, an agent using its tools, and an environment responding to those actions. Finally, it evaluates the agent's performance on two levels: whether it achieved the user's goal (the outcome) and whether its process for getting there was logical and safe (the execution path). By programmatically varying user and environment "personalities" (e.g., a buggy environment or an impatient user), we can efficiently explore a wide range of challenging, edge-case conditions.

### Key Contributions
1.  **A fully automated framework** for testing AI agents through end-to-end user and environment simulation, requiring only the agent's tool descriptions as input.
2.  **Novel user goal generation** that creates diverse and realistic test scenarios by first modeling tool relationships as a graph and then generating tasks based on logical paths within that graph.
3.  **A comprehensive evaluation suite** to measure both the simulation's validity and the agent's performance, including:
    *   Metrics to ensure simulated users and environments behave realistically.
    *   A `WeightedLevenshteinMetric` to assess the correctness and safety of the agent's tool-use sequence.
    *   A `GoalAchievementMetric` that uses an LLM-as-a-judge to evaluate if the final outcome satisfied the user's intent.

### Components of the system
The system has the following components:

1. **User Goal Generation (`src/afma/simulation/user.py`)**: This is the starting point of the simulation pipeline. Instead of manually writing test cases, this component automatically generates a diverse set of user goals based on the agent's capabilities (i.e., its tools) similarly to how fuzzing works for code testing.
    - **Tool Relationship Graph**: The process begins by creating a directed graph of tool relationships. An LLM analyzes the entire toolset and determines which tools logically follow others in realistic user workflows (e.g., `read_file` is often followed by `edit_file`). This creates a map of sensible tool sequences.
    - **Generating Scenarios**: By traversing this graph, the system generates "reasonable tool sequences" of varying lengths. For each sequence, an LLM then crafts a natural language **user goal** and a corresponding **initial environment state**. These two are kept separate to avoid biasing the environment.
    - **User Personalities**: The system can also assign "personalities" to users (e.g., "concise," "talkative," "german-speaking"). These personalities are instructions that modify the user simulation LLM's behavior, allowing the agent to be tested against various interaction styles.

2. **Simulated Environment (`src/afma/simulation/environment.py`)**: This component plays the role of the world the agent interacts with. While the framework can connect to real environments (e.g., a real filesystem or API), its power lies in simulation.
    - **LLM-Powered Simulation**: The `SimulatedEnvironment` uses an LLM to simulate the output of tool calls. It takes the tool name and arguments from the agent and asks an LLM to generate a realistic output.
    - **Stateful and Consistent**: The simulation is stateful. It maintains a history of all tool calls and their results within a session. This history is fed back into the simulation LLM for subsequent calls, ensuring consistency (e.g., a file created in one step can be read in a later step).
    - **Environment Personalities**: Similar to users, environments can have "personalities" to test agent robustness. These can simulate non-ideal conditions like a "buggy" environment that returns random errors, a "rate-limited" environment that fails intermittently, or even an "adversarial" one.

3. **User Simulation (`src/afma/simulation/user.py`)**: The `User` class drives the conversation with the agent.
    - **LLM-driven Dialogue**: The user is an LLM agent whose behavior is guided by a detailed system prompt. This prompt instructs the user to pursue their generated goal, to avoid accepting out-of-scope suggestions from the agent, and to be persistent in the face of errors.
    - **Goal Adherence**: This strict guidance ensures the simulation is a valid test of the agent's ability to complete a specific task, rather than a free-form chat. The user knows to terminate the conversation once the goal is met.

4. **Agent (`src/afma/simulation/agent.py`)**: The framework includes a standard, tool-calling `Agent`.
    - **Standard Architecture**: The agent follows a typical loop: it receives a user message, calls an LLM to decide on actions (i.e., tool calls), executes those calls against the environment (real or simulated), and then uses the results to either call more tools or formulate a response to the user.
    - **Environment-Agnostic**: The agent is designed to work with any environment that conforms to the `EnvironmentInterface`, meaning the same agent code can be run against a real or a simulated environment without modification.

### Simulation Reasonability and Evaluation
A key challenge in simulation is ensuring it is a meaningful proxy for reality. This framework includes several mechanisms and metrics to evaluate the simulation's validity and the agent's performance.

1. **User Goal Reasonability & Trace Alignment**: How do we know the generated user goals are realistic? We measure this by comparing the *planned* tool sequence (generated during user goal creation) with the *actual* tool sequence executed by the agent.
    - **`WeightedLevenshteinMetric` (`src/afma/evaluation/metrics/weighted_levenshtein.py`)**: This is a sophisticated alignment metric. It doesn't just count differences; it "weighs" them.
        - **Semantic Substitution Cost**: Substituting a tool with a semantically similar one (e.g., `create_file` for `touch_file`) has a low cost. The similarity is calculated using text embeddings of the tools' descriptions.
        - **Side-Effect Severity Cost**: Inserting an unexpected tool is penalized based on its potential for harm. A custom `SideEffectSeverityAssessor` uses an LLM to rate tools, so inserting a destructive tool like `delete_repository` is far more "expensive" than inserting `list_files`.
    - The final output is a similarity score (e.g., 96% alignment), which indicates how closely the agent's execution path matched the logical path of the user's goal.

2. **User Adherence to Goal**: We need to verify that the simulated user is behaving as instructed.
    - **LLM-as-a-Judge (`afma_testing/user_goal_adherence/run_user_goal_adherence_test.py`)**: A separate test pipeline extracts all user messages from simulated traces and uses an LLM to assess if the user adhered to their original goal without scope creep. Our experiments show ~99.6% adherence.

3. **Environment Simulation Fidelity**: Does the simulated environment behave like a real one?
    - **Real vs. Simulated Test (`afma_testing/environment/run_test.py`)**: We test this by running the same sequence of tool calls in a real environment (e.g., the local filesystem) and a simulated one. We then compare the final states of both environments (e.g., file contents). For reasonably complex scenarios, we observe high similarity (~90%), confirming the viability of the LLM-based simulation.

### Results and Goal Achievement
The ultimate measure of an agent's success is whether it accomplishes the user's goal.

- **`GoalAchievementMetric` (`src/afma/evaluation/metrics/goal_achievement.py`)**: This metric uses an LLM-as-a-judge to provide a final verdict. It looks at the initial environment state, the user's goal, and the final environment state after the simulation and determines to what extent the goal was achieved. This is an outcome-focused metric; it doesn't care *how* the agent did it, only *if* it succeeded. The result is a score from 0.0 to 1.0.

By combining these components, this framework provides a powerful, automated way to analyze agent failure modes across a wide range of tasks and conditions before deployment.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/afma.git
cd afma

# Install dependencies using uv
uv sync
```

## References
1. [ToolFuzz](https://arxiv.org/pdf/2503.04479)
2. [ToolEmu](https://arxiv.org/pdf/2309.15817)
3. [AgentMonitor](https://github.com/chanchimin/AgentMonitor)
4. [Athena](https://arxiv.org/pdf/2408.11021)
5. [Testing and Understanding Erroneous Planning in LLM Agents through Synthesized User Inputs](https://arxiv.org/abs/2404.17833)
6. [What Limits LLM-based Human Simulation: LLMs or Our Design?](https://arxiv.org/abs/2501.08579)
7. [Blogpost about problems in coding agents](https://ezyang.github.io/ai-blindspots/)
8. [GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts](https://arxiv.org/pdf/2309.10253)
9. [SWE-Agent includes some failure mode analysis (e.g. p8)](https://arxiv.org/pdf/2405.15793)
10. [Tau-bench](https://arxiv.org/pdf/2406.12045)
11. [MCP Servers in the MCP orgs](https://github.com/modelcontextprotocol/servers/tree/main/src)
12. [Agentdojo](https://arxiv.org/abs/2406.13352)
13. [Firewalls to Secure Dynamic LLM Agentic Networks](https://arxiv.org/pdf/2502.01822)
14. [HAICOSYSTEM](https://arxiv.org/pdf/2409.16427)
15. [ADELE](https://arxiv.org/pdf/2503.06378) / [Website](https://kinds-of-intelligence-cfi.github.io/ADELE/)
16. [CodeAct](https://arxiv.org/pdf/2402.01030) (similar: [Alternative paper](https://arxiv.org/pdf/2312.04511))
17. [TravelPlanner](https://arxiv.org/pdf/2402.01622)
18. [WebCanvas - key nodes idea](https://arxiv.org/pdf/2406.12373)
19. [M&M's](https://arxiv.org/pdf/2403.11085)
20. [APIGen: Automated PIpeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/pdf/2406.18518)
21. [Agent-as-a-judge](https://arxiv.org/abs/2410.10934)
22. [Camel](https://arxiv.org/pdf/2503.18813) (some security thingy)
23. [Multi-agent failure modes](https://arxiv.org/pdf/2503.13657)
24. [Interactive Debugging and Steering of Multi-Agent AI Systems](https://dl.acm.org/doi/pdf/10.1145/3706598.3713581)
25. [LLM usage distribution across industries](https://www.anthropic.com/news/the-anthropic-economic-index)
26. [Terminal-Bench](https://www.tbench.ai/about)
27. [Darwin Gödel Machine:
Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/pdf/2505.22954)
28. [Estimating Correctness Without Oracles in LLM-Based Code Generation](https://arxiv.org/pdf/2507.00057)

## Notes
Always use uv and pyproject.toml for package and dependencies management.