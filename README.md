# Agent Failure Mode Analysis through User Simulation

## Overview
AI agents are becoming increasingly capable, yet remain error-prone and challenging to trust in real-world high-stakes applications. They often misinterpret user intent, behave unpredictably in complex workflows, and fail under ambiguous task specifications. Without a structured understanding of these failure modes, production deployment remains risky and ad hoc. In this project, we aim to provide this structured understanding via a systematic testing and simulation approach. Knowing the tools available to an agent, we can simulate both user input and tool output that explores error-prone and unsafe edge-cases.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/afma.git
cd afma

# Install dependencies using uv
uv sync
```

## How to Use

The `afma` command-line tool provides several utilities for agent testing and simulation:

### Scanning Agent Tool Configurations

To scan an MCP agent configuration and extract its toolset:

```bash
afma scan-mcp agent_mcp_configs/claude_desktop_config.json -o results/toolset.json
```

This command analyzes the agent configuration and outputs the detected toolset to a JSON file.

### Creating Simulated User Inputs

Generate simulated user inputs for testing:

```bash
afma create-users agent_mcp_configs/claude_desktop_config.json litellm_configs/o4-mini.json results/users.json -m 3 --max-users-per-len 10
```

This command:
- Uses the provided agent configuration and LiteLLM config
- Simulates user tasks with maximum tool sequence length of 3
- Creates up to 10 users per tool sequence length
- Saves the results to `results/users.json`

## This Project
In this project, we plan to address that gap by simulating diverse user-agent interactions to uncover, categorize, and evaluate agent failures systematically. We focus on building the foundations for scalable, automated failure mode analysis - reducing reliance on manual testing or reactive fixes after deployment. Concretely, we focus on four key objectives:

- **Identify Failure Modes**: Uncover common and edge-case agent errors, such as tool misuse, misinterpretation, and unstable behavior across varying task conditions [5, 9].
- **Measure Robustness**: Analyze how agent reliability changes as tasks grow in complexity or ambiguity, drawing from ideas in [4, 10].
- **Simulate High-Risk Behavior Safely**: Detect potentially harmful behavior (e.g., executing dangerous commands) through controlled simulations inspired by [1, 2, 8].
- **Provide a Scalable Testing Framework**: Build a foundation for automated agent testing and auditing, supporting safe iteration in development [3, 7, 12].

To enable the comprehensive evaluation, we plan to follow the approach in the figure below. As a first step, we plan to build a User Simulation System (blue, yellow in the figure) that generates tool-aware user requests aiming to trigger the agent to execute predefined tool interactions and sequences ([5, 10]). Building on this, we plan to make the user simulation variable in user specificity, task structure, ambiguity, and other (possible) failure causes. We aim to quantify the overall agent robustness over these factors and task complexities [4, 6, 9]. To enable testing of dangerous behaviors (e.g., tools with destructive impact on the environment like rm, spending money, sending email to real addresses), we plan to introduce a simulation layer for such tools [2]. On the implementation side, we plan to abstract this behavior behind an MCP server [11], which can multiplex between the actual and simulated tools.

![Project approach diagram](illustration.png)

## Milestones
- Ramp-up and Reading (0.5 months)
- Core Technical Work (2.5 months)
  - Agent-Specific User Simulation
    - for fixed agents, simulate user-flows that reach various tool interactions
  - User Simulation with Scenario Diversity
    - make user input generation work for various agents
    - introduce nuance, variation and ambiguity to tasks and user requests
  - Safe Simulation of Tools
    - use LLM-driven simulation of unsafe tools
  - Large-scale evaluation
    - Quantitatively compare agent outputs with ground-truth traces
    - Identify robustness gaps and security vulnerabilities
    - Write-up, potential paper submission, open benchmark (1 month)

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

## Notes
Always use uv and pyproject.toml for package and dependencies management.