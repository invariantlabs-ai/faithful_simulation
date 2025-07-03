# User Generation Testing Module

This module provides comprehensive testing for the user generation process in AFMA. It assesses the correctness and quality of generated user goals and environment expectations.

## Overview

The user generation testing module evaluates two critical aspects of user generation:

1. **Environment Completeness**: Does the environment expectations contain all necessary information for the user to achieve their goal?
2. **Goal Unambiguity**: Does the user goal and environment expectations contain all relevant information for the original tool sequence?

## Components

### 1. EnvironmentCompletenessAssessor
- Evaluates if environment expectations are sufficient
- Uses LLM to assess completeness with detailed reasoning
- Provides scores and missing elements analysis

### 2. GoalUnambiguityAssessor
- Evaluates if user goals contain all necessary information
- Compares goals against original tool sequences
- Assesses parameter completeness and clarity

### 3. UserGenerationTester
- Orchestrates the entire testing process
- Generates users using CombinatoricUserSet
- Combines results and generates comprehensive reports
- Saves detailed results and summaries

## Configuration

The module uses a YAML configuration file with the following structure:

```yaml
# Path to MCP configuration file
mcp_config_path: "agent_mcp_configs/files.json"

user_generation_testing:
  # LLM configuration for user generation
  generation_litellm:
    model: gpt-4.1-mini
    temperature: 0.3
    top_p: 0.9
    timeout: 30
    caching: true
  
  # LLM configuration for assessment
  assessment_litellm:
    model: gpt-4.1-mini
    temperature: 0.1
    top_p: 0.9
    timeout: 30
    caching: true
  
  # Test parameters
  test:
    max_concurrent_users: 10
    results_output_dir: "results/user_generation_testing"
    
  # User generation parameters
  user_generation:
    permutation_lengths: [1, 2, 3]
    max_users_per_length: 5
    semaphore_limit: 10
    
  # Assessment parameters
  assessment:
    environment_completeness:
      enabled: true
      scoring_threshold: 0.7
    goal_unambiguity:
      enabled: true
      scoring_threshold: 0.8
    concurrency: 5
```

## Usage

### Command Line Interface

Run the testing using the provided script:

```bash
cd testing/user_generation
python run_user_generation_test.py config.yaml
```

### Programmatic Usage

```python
import asyncio
from testing.user_generation.user_generation_tester import UserGenerationTester

async def run_test():
    tester = UserGenerationTester("config.yaml")
    summary = await tester.run_test()
    return summary

# Run the test
summary = asyncio.run(run_test())
```

## Output

The module generates several output files in the specified results directory:

### 1. detailed_results.json
Contains detailed assessment results for each user, including:
- User goals and environment expectations
- Tool sequences
- Environment completeness scores and reasoning
- Goal unambiguity scores and reasoning
- Missing elements and information

### 2. summary.json
Contains aggregated statistics:
- Average scores for both assessments
- Score distributions
- Tool sequence length breakdowns

### 3. config.yaml
Copy of the configuration used for the test run

## Assessment Criteria

### Environment Completeness (0.0-1.0)
- **0.9-1.0**: Environment is complete, no missing elements
- **0.7-0.8**: Environment is mostly complete, minor missing elements
- **0.5-0.6**: Environment is partially complete, some important missing elements
- **0.3-0.4**: Environment is incomplete, significant missing elements
- **0.1-0.2**: Environment is very incomplete, critical missing elements

### Goal Unambiguity (0.0-1.0)
- **0.9-1.0**: Goal is completely unambiguous, all information provided
- **0.7-0.8**: Goal is mostly unambiguous, minor missing details
- **0.5-0.6**: Goal is partially unambiguous, some important missing information
- **0.3-0.4**: Goal is ambiguous, significant missing information
- **0.1-0.2**: Goal is very ambiguous, critical missing information

## Dependencies

- `litellm`: For LLM interactions with caching
- `loguru`: For logging
- `pyyaml`: For configuration parsing
- `asyncio`: For concurrent operations

## Integration with AFMA

This module integrates with the existing AFMA codebase:
- Uses the same MCP configuration loading mechanism as `cli.py`
- Leverages the `CombinatoricUserSet` from `user.py`
- Follows the same patterns as the environment testing module
- Uses consistent logging and error handling patterns 