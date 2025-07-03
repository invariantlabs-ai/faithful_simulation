import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from loguru import logger
import yaml

from src.afma.simulation.user import CombinatoricUserSet, User
from src.afma.simulation.utils import get_tools_from_mcp_config
from .environment_completeness_assessor import EnvironmentCompletenessAssessor, EnvironmentCompletenessResult
from .goal_unambiguity_assessor import GoalUnambiguityAssessor, GoalUnambiguityResult


@dataclass
class UserGenerationTestResult:
    """Result of user generation testing for a single user."""
    user_index: int
    user_goal: str
    environment_expectations: str
    tool_sequence: List[Dict[str, Any]]
    
    # Assessment results
    environment_completeness: EnvironmentCompletenessResult
    goal_unambiguity: GoalUnambiguityResult
    
    # Metadata
    tool_sequence_length: int
    assessment_timestamp: str


@dataclass
class UserGenerationTestSummary:
    """Summary of all user generation test results."""
    total_users: int
    avg_environment_completeness: float
    avg_goal_unambiguity: float
    environment_completeness_distribution: Dict[str, int]
    goal_unambiguity_distribution: Dict[str, int]
    tool_sequence_length_breakdown: Dict[int, int]
    
    # Average metrics by sequence length
    avg_metrics_by_length: Dict[int, Dict[str, float]]
    
    # Detailed results
    results: List[UserGenerationTestResult]
    
    # Configuration
    config: Dict[str, Any]


class UserGenerationTester:
    """Main tester for user generation correctness."""
    
    def __init__(self, config_path: str):
        """
        Initialize the user generation tester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize assessors
        self.environment_assessor = EnvironmentCompletenessAssessor(
            llm_config=self.config["user_generation_testing"]["assessment_litellm"],
            concurrency=self.config["user_generation_testing"]["assessment"]["concurrency"]
        )
        
        self.goal_assessor = GoalUnambiguityAssessor(
            llm_config=self.config["user_generation_testing"]["assessment_litellm"],
            concurrency=self.config["user_generation_testing"]["assessment"]["concurrency"]
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = [
            "mcp_config_path",
            "user_generation_testing.generation_litellm",
            "user_generation_testing.assessment_litellm",
            "user_generation_testing.user_generation.permutation_lengths",
            "user_generation_testing.assessment.concurrency"
        ]
        
        for field in required_fields:
            keys = field.split('.')
            current = config
            for key in keys:
                if key not in current:
                    raise ValueError(f"Missing required config field: {field}")
                current = current[key]
        
        return config
    
    async def _generate_users(self) -> List[User]:
        """Generate users using CombinatoricUserSet."""
        # Load tools from MCP config
        logger.info(f"Loading tools from MCP config: {self.config['mcp_config_path']}")
        tools = await get_tools_from_mcp_config(
            self.config["mcp_config_path"], 
            self.config["user_generation_testing"]["generation_litellm"].get("timeout", 30)
        )
        logger.success(f"Loaded {len(tools)} tools from MCP config")
        
        # Create user set
        user_set = CombinatoricUserSet(
            tools_info=tools,
            generation_llm_config=self.config["user_generation_testing"]["generation_litellm"],
            simulation_llm_config=self.config["user_generation_testing"]["generation_litellm"],
            permutation_lengths=self.config["user_generation_testing"]["user_generation"]["permutation_lengths"],
            max_users_per_len=self.config["user_generation_testing"]["user_generation"]["max_users_per_length"],
            semaphore_limit=self.config["user_generation_testing"]["user_generation"]["semaphore_limit"]
        )
        
        logger.info(f"Generating users with lengths {self.config['user_generation_testing']['user_generation']['permutation_lengths']}")
        users = await user_set.generate_users()
        logger.success(f"Generated {len(users)} users")
        
        return users
    
    async def run_test(self) -> UserGenerationTestSummary:
        """
        Run the complete user generation test.
        
        Returns:
            UserGenerationTestSummary with all results
        """
        logger.info("Starting user generation testing")
        
        # Step 1: Generate users
        logger.info("Step 1: Generating users")
        users = await self._generate_users()
        
        # Step 2: Convert users to data format for assessment
        user_data = []
        for user in users:
            user_data.append({
                "user_goal": user.user_goal,
                "environment_expectations": user.environment_expectations,
                "source": user.source
            })
        
        # Step 3: Assess environment completeness
        logger.info("Step 2: Assessing environment completeness")
        environment_results = await self.environment_assessor.assess_batch(user_data)
        
        # Step 4: Assess goal unambiguity
        logger.info("Step 3: Assessing goal unambiguity")
        goal_results = await self.goal_assessor.assess_batch(user_data)
        
        # Step 5: Combine results
        logger.info("Step 4: Combining results")
        test_results = []
        for i, user in enumerate(users):
            result = UserGenerationTestResult(
                user_index=i,
                user_goal=user.user_goal,
                environment_expectations=user.environment_expectations,
                tool_sequence=user.source,
                environment_completeness=environment_results[i],
                goal_unambiguity=goal_results[i],
                tool_sequence_length=len(user.source),
                assessment_timestamp=str(asyncio.get_event_loop().time())
            )
            test_results.append(result)
        
        # Step 6: Create summary
        logger.info("Step 5: Creating summary")
        summary = self._create_summary(test_results)
        
        # Step 7: Save results
        logger.info("Step 6: Saving results")
        await self._save_results(test_results, summary)
        
        logger.success("User generation testing completed")
        return summary
    
    def _create_summary(self, results: List[UserGenerationTestResult]) -> UserGenerationTestSummary:
        """Create summary statistics from test results."""
        total_users = len(results)
        
        # Calculate averages
        env_scores = [r.environment_completeness.score for r in results if r.environment_completeness.error is None]
        goal_scores = [r.goal_unambiguity.score for r in results if r.goal_unambiguity.error is None]
        
        avg_env_completeness = sum(env_scores) / len(env_scores) if env_scores else 0.0
        avg_goal_unambiguity = sum(goal_scores) / len(goal_scores) if goal_scores else 0.0
        
        # Create distributions
        env_distribution = self._create_score_distribution([r.environment_completeness.score for r in results])
        goal_distribution = self._create_score_distribution([r.goal_unambiguity.score for r in results])
        
        # Tool sequence length breakdown
        length_breakdown = {}
        for result in results:
            length = result.tool_sequence_length
            length_breakdown[length] = length_breakdown.get(length, 0) + 1
        
        # Average metrics by sequence length
        avg_metrics_by_length = {}
        for length in length_breakdown:
            length_results = [r for r in results if r.tool_sequence_length == length]
            if length_results:
                env_scores = [r.environment_completeness.score for r in length_results if r.environment_completeness.error is None]
                goal_scores = [r.goal_unambiguity.score for r in length_results if r.goal_unambiguity.error is None]
                
                avg_env_completeness = sum(env_scores) / len(env_scores) if env_scores else 0.0
                avg_goal_unambiguity = sum(goal_scores) / len(goal_scores) if goal_scores else 0.0
                
                avg_metrics_by_length[length] = {
                    "avg_environment_completeness": avg_env_completeness,
                    "avg_goal_unambiguity": avg_goal_unambiguity
                }
        
        return UserGenerationTestSummary(
            total_users=total_users,
            avg_environment_completeness=avg_env_completeness,
            avg_goal_unambiguity=avg_goal_unambiguity,
            environment_completeness_distribution=env_distribution,
            goal_unambiguity_distribution=goal_distribution,
            tool_sequence_length_breakdown=length_breakdown,
            avg_metrics_by_length=avg_metrics_by_length,
            results=results,
            config=self.config
        )
    
    def _create_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Create distribution of scores into categories."""
        distribution = {
            "excellent": 0,  # 0.0-0.2 (very few issues)
            "good": 0,       # 0.2-0.4 (few issues)
            "fair": 0,       # 0.4-0.6 (moderate issues)
            "poor": 0,       # 0.6-0.8 (many issues)
            "very_poor": 0   # 0.8+ (lots of issues)
        }
        
        for score in scores:
            if score <= 0.2:
                distribution["excellent"] += 1
            elif score <= 0.4:
                distribution["good"] += 1
            elif score <= 0.6:
                distribution["fair"] += 1
            elif score <= 0.8:
                distribution["poor"] += 1
            else:
                distribution["very_poor"] += 1
        
        return distribution
    
    async def _save_results(self, results: List[UserGenerationTestResult], summary: UserGenerationTestSummary):
        """Save test results to files."""
        output_dir = Path(self.config["user_generation_testing"]["test"]["results_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "detailed_results.json"
        results_data = [asdict(result) for result in results]
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary
        summary_file = output_dir / "summary.json"
        summary_dict = asdict(summary)
        summary_dict["results"] = [asdict(r) for r in summary.results]
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        # Save configuration
        config_file = output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.success(f"Results saved to {output_dir}")
        
        # Print summary to console
        self._print_summary(summary)
    
    def _print_summary(self, summary: UserGenerationTestSummary):
        """Print summary to console."""
        logger.info("=" * 60)
        logger.info("USER GENERATION TESTING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total users tested: {summary.total_users}")
        logger.info(f"Average environment completeness issues: {summary.avg_environment_completeness:.3f} (issues/tool)")
        logger.info(f"Average goal unambiguity issues: {summary.avg_goal_unambiguity:.3f} (issues/tool)")
        
        logger.info("\nEnvironment Completeness Distribution (lower is better):")
        for category, count in summary.environment_completeness_distribution.items():
            percentage = (count / summary.total_users) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        logger.info("\nGoal Unambiguity Distribution (lower is better):")
        for category, count in summary.goal_unambiguity_distribution.items():
            percentage = (count / summary.total_users) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        logger.info("\nTool Sequence Length Breakdown:")
        for length, count in summary.tool_sequence_length_breakdown.items():
            percentage = (count / summary.total_users) * 100
            logger.info(f"  {length} tools: {count} ({percentage:.1f}%)")
        
        logger.info("\nAverage Issues by Sequence Length (lower is better):")
        for length in sorted(summary.avg_metrics_by_length.keys()):
            metrics = summary.avg_metrics_by_length[length]
            count = summary.tool_sequence_length_breakdown[length]
            logger.info(f"  {length} tools ({count} users):")
            logger.info(f"    Environment Completeness issues: {metrics['avg_environment_completeness']:.3f}")
            logger.info(f"    Goal Unambiguity issues: {metrics['avg_goal_unambiguity']:.3f}")
        
        logger.info("=" * 60) 