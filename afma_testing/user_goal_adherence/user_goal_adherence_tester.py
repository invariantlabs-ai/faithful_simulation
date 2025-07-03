import asyncio
import json
import os
import statistics
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from loguru import logger
import yaml

from .user_goal_adherence_assessor import UserGoalAdherenceAssessor, UserGoalAdherenceResult


@dataclass
class UserGoalAdherenceTestResult:
    """Result of user goal adherence testing for a single conversation."""
    conversation_id: int
    user_goal: str
    user_personality: Optional[str]
    user_personality_name: Optional[str]
    environment_personality: Optional[str]
    environment_personality_name: Optional[str]
    trace_set_id: Optional[str]
    instantiation_id: Optional[int]
    
    # Assessment results
    message_results: List[UserGoalAdherenceResult]
    
    # Aggregated metrics
    total_messages: int
    adhering_messages: int
    non_adhering_messages: int
    adherence_rate: float
    avg_adherence_score: float
    
    # Metadata
    assessment_timestamp: str


@dataclass
class UserGoalAdherenceTestSummary:
    """Summary of all user goal adherence test results."""
    total_conversations: int
    total_messages: int
    overall_adherence_rate: float
    overall_avg_score: float
    overall_std_score: float
    
    # Distribution statistics
    adherence_rate_distribution: Dict[str, int]
    score_distribution: Dict[str, int]
    
    # Results by personality
    results_by_user_personality: Dict[str, Dict[str, float]]
    results_by_environment_personality: Dict[str, Dict[str, float]]
    
    # Detailed results
    results: List[UserGoalAdherenceTestResult]
    
    # Configuration
    config: Dict[str, Any]


class UserGoalAdherenceTester:
    """Main tester for user goal adherence evaluation."""
    
    def __init__(self, config_path: str):
        """
        Initialize the user goal adherence tester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize assessor
        self.assessor = UserGoalAdherenceAssessor(
            llm_config=self.config["user_goal_adherence_testing"]["assessment_litellm"],
            concurrency=self.config["user_goal_adherence_testing"]["assessment"]["concurrency"]
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = [
            "conversations_path",
            "user_goal_adherence_testing.assessment_litellm",
            "user_goal_adherence_testing.assessment.concurrency"
        ]
        
        for field in required_fields:
            keys = field.split('.')
            current = config
            for key in keys:
                if key not in current:
                    raise ValueError(f"Missing required config field: {field}")
                current = current[key]
        
        return config
    
    def _load_conversations(self) -> List[Dict[str, Any]]:
        """Load conversations from the specified file."""
        conversations_path = self.config["conversations_path"]
        
        if not os.path.exists(conversations_path):
            raise FileNotFoundError(f"Conversations file not found: {conversations_path}")
        
        logger.info(f"Loading conversations from {conversations_path}")
        with open(conversations_path, 'r') as f:
            conversations = json.load(f)
        
        logger.success(f"Loaded {len(conversations)} conversations")
        return conversations
    
    async def run_test(self) -> UserGoalAdherenceTestSummary:
        """
        Run the complete user goal adherence test.
        
        Returns:
            UserGoalAdherenceTestSummary with all results
        """
        logger.info("Starting user goal adherence testing")
        
        # Step 1: Load conversations
        logger.info("Step 1: Loading conversations")
        conversations = self._load_conversations()
        
        # Step 2: Assess all conversations
        logger.info("Step 2: Assessing conversations for goal adherence")
        all_message_results = await self.assessor.assess_batch(conversations)
        
        # Step 3: Process results
        logger.info("Step 3: Processing results")
        test_results = []
        for i, (conversation, message_results) in enumerate(zip(conversations, all_message_results)):
            result = self._process_conversation_results(i, conversation, message_results)
            test_results.append(result)
        
        # Step 4: Create summary
        logger.info("Step 4: Creating summary")
        summary = self._create_summary(test_results)
        
        # Step 5: Save results
        logger.info("Step 5: Saving results")
        await self._save_results(test_results, summary)
        
        logger.success("User goal adherence testing completed")
        return summary
    
    def _process_conversation_results(self, conversation_id: int, conversation: Dict[str, Any], 
                                    message_results: List[UserGoalAdherenceResult]) -> UserGoalAdherenceTestResult:
        """Process results for a single conversation."""
        
        # Calculate aggregated metrics
        total_messages = len(message_results)
        adhering_messages = sum(1 for r in message_results if r.score == 1.0)
        non_adhering_messages = total_messages - adhering_messages
        adherence_rate = adhering_messages / total_messages if total_messages > 0 else 0.0
        
        # Calculate average score (excluding errors)
        valid_scores = [r.score for r in message_results if r.error is None]
        avg_adherence_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        return UserGoalAdherenceTestResult(
            conversation_id=conversation_id,
            user_goal=conversation["user_goal"],
            user_personality=conversation.get("user_personality"),
            user_personality_name=conversation.get("user_personality_name"),
            environment_personality=conversation.get("environment_personality"),
            environment_personality_name=conversation.get("environment_personality_name"),
            trace_set_id=conversation.get("trace_set_id"),
            instantiation_id=conversation.get("instantiation_id"),
            message_results=message_results,
            total_messages=total_messages,
            adhering_messages=adhering_messages,
            non_adhering_messages=non_adhering_messages,
            adherence_rate=adherence_rate,
            avg_adherence_score=avg_adherence_score,
            assessment_timestamp=str(asyncio.get_event_loop().time())
        )
    
    def _create_summary(self, results: List[UserGoalAdherenceTestResult]) -> UserGoalAdherenceTestSummary:
        """Create summary statistics from test results."""
        total_conversations = len(results)
        
        # Calculate overall metrics
        all_scores = []
        all_adherence_rates = []
        total_messages = 0
        
        for result in results:
            total_messages += result.total_messages
            all_adherence_rates.append(result.adherence_rate)
            # Add individual message scores
            for msg_result in result.message_results:
                if msg_result.error is None:
                    all_scores.append(msg_result.score)
        
        overall_adherence_rate = sum(all_adherence_rates) / len(all_adherence_rates) if all_adherence_rates else 0.0
        overall_avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        overall_std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        
        # Create distributions
        adherence_rate_distribution = self._create_rate_distribution(all_adherence_rates)
        score_distribution = self._create_score_distribution(all_scores)
        
        # Results by personality
        results_by_user_personality = self._aggregate_by_personality(results, "user_personality_name")
        results_by_environment_personality = self._aggregate_by_personality(results, "environment_personality_name")
        
        return UserGoalAdherenceTestSummary(
            total_conversations=total_conversations,
            total_messages=total_messages,
            overall_adherence_rate=overall_adherence_rate,
            overall_avg_score=overall_avg_score,
            overall_std_score=overall_std_score,
            adherence_rate_distribution=adherence_rate_distribution,
            score_distribution=score_distribution,
            results_by_user_personality=results_by_user_personality,
            results_by_environment_personality=results_by_environment_personality,
            results=results,
            config=self.config
        )
    
    def _create_rate_distribution(self, rates: List[float]) -> Dict[str, int]:
        """Create distribution of adherence rates."""
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for rate in rates:
            if rate < 0.2:
                distribution["0.0-0.2"] += 1
            elif rate < 0.4:
                distribution["0.2-0.4"] += 1
            elif rate < 0.6:
                distribution["0.4-0.6"] += 1
            elif rate < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1
        
        return distribution
    
    def _create_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Create distribution of individual message scores."""
        distribution = {
            "0.0": 0,
            "1.0": 0
        }
        
        for score in scores:
            if score == 0.0:
                distribution["0.0"] += 1
            elif score == 1.0:
                distribution["1.0"] += 1
        
        return distribution
    
    def _aggregate_by_personality(self, results: List[UserGoalAdherenceTestResult], 
                                personality_field: str) -> Dict[str, Dict[str, float]]:
        """Aggregate results by personality type."""
        personality_groups = {}
        
        for result in results:
            personality = getattr(result, personality_field) or "default"
            
            if personality not in personality_groups:
                personality_groups[personality] = {
                    "count": 0,
                    "total_adherence_rate": 0.0,
                    "total_avg_score": 0.0
                }
            
            personality_groups[personality]["count"] += 1
            personality_groups[personality]["total_adherence_rate"] += result.adherence_rate
            personality_groups[personality]["total_avg_score"] += result.avg_adherence_score
        
        # Calculate averages
        aggregated = {}
        for personality, data in personality_groups.items():
            count = data["count"]
            aggregated[personality] = {
                "count": count,
                "avg_adherence_rate": data["total_adherence_rate"] / count,
                "avg_score": data["total_avg_score"] / count
            }
        
        return aggregated
    
    async def _save_results(self, results: List[UserGoalAdherenceTestResult], 
                          summary: UserGoalAdherenceTestSummary):
        """Save test results to files."""
        output_dir = self.config["user_goal_adherence_testing"]["test"]["results_output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_results = []
        for result in results:
            detailed_result = asdict(result)
            # Convert message results to dict format
            detailed_result["message_results"] = [asdict(msg) for msg in result.message_results]
            detailed_results.append(detailed_result)
        
        detailed_path = os.path.join(output_dir, "detailed_results.json")
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.success(f"Saved detailed results to {detailed_path}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        logger.success(f"Saved summary to {summary_path}")
        
        # Print summary
        self._print_summary(summary)
    
    def _print_summary(self, summary: UserGoalAdherenceTestSummary):
        """Print a human-readable summary of results."""
        logger.info("\n" + "="*60)
        logger.info("USER GOAL ADHERENCE TESTING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total conversations: {summary.total_conversations}")
        logger.info(f"Total messages: {summary.total_messages}")
        logger.info(f"Overall adherence rate: {summary.overall_adherence_rate:.3f}")
        logger.info(f"Overall average score: {summary.overall_avg_score:.3f}")
        logger.info(f"Overall score std dev: {summary.overall_std_score:.3f}")
        
        logger.info("\nAdherence Rate Distribution:")
        for range_name, count in summary.adherence_rate_distribution.items():
            logger.info(f"  {range_name}: {count} conversations")
        
        logger.info("\nScore Distribution:")
        for score, count in summary.score_distribution.items():
            logger.info(f"  Score {score}: {count} messages")
        
        if summary.results_by_user_personality:
            logger.info("\nResults by User Personality:")
            for personality, data in summary.results_by_user_personality.items():
                logger.info(f"  {personality}: {data['count']} convs, "
                          f"avg rate: {data['avg_adherence_rate']:.3f}, "
                          f"avg score: {data['avg_score']:.3f}")
        
        if summary.results_by_environment_personality:
            logger.info("\nResults by Environment Personality:")
            for personality, data in summary.results_by_environment_personality.items():
                logger.info(f"  {personality}: {data['count']} convs, "
                          f"avg rate: {data['avg_adherence_rate']:.3f}, "
                          f"avg score: {data['avg_score']:.3f}")
        
        logger.info("="*60) 