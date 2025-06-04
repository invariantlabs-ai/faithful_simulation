from typing import Any, Dict, List, Optional
import asyncio
from dataclasses import dataclass, asdict
from loguru import logger

from .metrics import EvaluationMetric, EvaluationResult, WeightedLevenshteinMetric


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across all conversations."""
    total_conversations: int
    metric_scores: Dict[str, float]  # metric_name -> average_score
    metric_details: Dict[str, List[Dict[str, Any]]]  # metric_name -> list of detailed results
    personality_breakdown: Optional[Dict[str, Dict[str, float]]] = None  # personality -> metric -> score
    environment_breakdown: Optional[Dict[str, Dict[str, float]]] = None  # environment -> metric -> score
    permutation_length_breakdown: Optional[Dict[int, Dict[str, float]]] = None  # length -> metric -> score
    combined_breakdown: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None  # user_personality -> env_personality -> metric -> score


class Evaluator:
    """Main evaluator class that orchestrates evaluation of conversation traces."""
    
    def __init__(self, 
                 tool_definitions: Dict[str, Dict[str, Any]],
                 embedding_config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            tool_definitions: Tool definitions for semantic similarity
            embedding_config: Embedding configuration for WeightedLevenshteinMetric
        """
        self.tool_definitions = tool_definitions
        self.embedding_config = embedding_config
        self.metrics: List[EvaluationMetric] = []
        
        # Initialize default metrics
        self.add_metric(WeightedLevenshteinMetric(
            tool_definitions=tool_definitions,
            embedding_config=embedding_config,
            cache_embeddings=True
        ))
        logger.info("Added WeightedLevenshteinMetric to evaluator")
    
    def add_metric(self, metric: EvaluationMetric) -> None:
        """Add an evaluation metric."""
        self.metrics.append(metric)
    
    async def evaluate_conversation(self, 
                                   user_goal: str,
                                   user_source: List[Dict[str, str]],
                                   execution_trace: List[Dict[str, Any]],
                                   used_tools: List[str],
                                   user_personality: Optional[str] = None,
                                   environment_personality: Optional[str] = None) -> Dict[str, EvaluationResult]:
        """
        Evaluate a single conversation using all configured metrics.
        
        Args:
            user_goal: The original user goal/request
            user_source: The expected tool sequence from user generation
            execution_trace: Full conversation history including tool calls
            used_tools: List of tool names that were actually used
            user_personality: Optional user personality type
            environment_personality: Optional environment personality type
            
        Returns:
            Dictionary mapping metric names to evaluation results
        """
        results = {}
        
        # Run all metrics concurrently
        tasks = []
        for metric in self.metrics:
            task = metric.evaluate(user_goal, user_source, execution_trace, used_tools)
            tasks.append((metric.metric_name, task))
        
        # Wait for all evaluations to complete
        for metric_name, task in tasks:
            try:
                result = await task
                results[metric_name] = result
                logger.debug(f"Metric {metric_name}: score={result.score:.3f}")
            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {e}")
                results[metric_name] = EvaluationResult(
                    metric_name=metric_name,
                    score=0.0,
                    error=str(e)
                )
        
        return results
    
    async def evaluate_batch(self, 
                            conversations: List[Dict[str, Any]],
                            concurrency: int = 5) -> EvaluationSummary:
        """
        Evaluate a batch of conversations.
        
        Args:
            conversations: List of conversation dictionaries with keys:
                - user_goal: str
                - user_source: List[Dict[str, str]]
                - history: List[Dict[str, Any]]
                - used_tools: List[str]
                - user_personality: Optional[str]
                - environment_personality: Optional[str]
            concurrency: Maximum number of concurrent evaluations
            
        Returns:
            EvaluationSummary with aggregated results
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def evaluate_single(conv_data: Dict[str, Any]) -> Dict[str, EvaluationResult]:
            async with semaphore:
                return await self.evaluate_conversation(
                    user_goal=conv_data["user_goal"],
                    user_source=conv_data["user_source"],
                    execution_trace=conv_data["history"],
                    used_tools=conv_data["used_tools"],
                    user_personality=conv_data.get("user_personality"),
                    environment_personality=conv_data.get("environment_personality")
                )
        
        logger.info(f"Evaluating {len(conversations)} conversations with concurrency {concurrency}")
        
        # Run all evaluations
        tasks = [evaluate_single(conv) for conv in conversations]
        all_results = await asyncio.gather(*tasks)
        
        # Aggregate results
        return self._aggregate_results(conversations, all_results)
    
    def _aggregate_results(self, 
                          conversations: List[Dict[str, Any]], 
                          all_results: List[Dict[str, EvaluationResult]]) -> EvaluationSummary:
        """Aggregate evaluation results into a summary."""
        
        # Initialize aggregation structures
        metric_scores = {}
        metric_details = {}
        personality_breakdown = {}
        environment_breakdown = {}
        permutation_length_breakdown = {}
        combined_breakdown = {}
        
        # Get all metric names
        if all_results:
            metric_names = list(all_results[0].keys())
        else:
            metric_names = []
        
        # Initialize metric aggregation
        for metric_name in metric_names:
            metric_scores[metric_name] = 0.0
            metric_details[metric_name] = []
        
        # Process each conversation result
        for i, (conv_data, results) in enumerate(zip(conversations, all_results)):
            user_personality = conv_data.get("user_personality", "default")
            env_personality = conv_data.get("environment_personality", "default")
            permutation_length = len(conv_data.get("user_source", []))
            
            # Initialize breakdowns if needed
            if user_personality not in personality_breakdown:
                personality_breakdown[user_personality] = {name: [] for name in metric_names}
            if env_personality not in environment_breakdown:
                environment_breakdown[env_personality] = {name: [] for name in metric_names}
            if permutation_length not in permutation_length_breakdown:
                permutation_length_breakdown[permutation_length] = {name: [] for name in metric_names}
            if user_personality not in combined_breakdown:
                combined_breakdown[user_personality] = {}
            if env_personality not in combined_breakdown[user_personality]:
                combined_breakdown[user_personality][env_personality] = {name: [] for name in metric_names}
            
            # Aggregate scores for each metric
            for metric_name, result in results.items():
                score = result.score if result.error is None else 0.0
                
                # Overall aggregation
                metric_scores[metric_name] += score
                metric_details[metric_name].append({
                    "conversation_id": i,
                    "score": score,
                    "details": result.details,
                    "error": result.error,
                    "user_personality": user_personality,
                    "environment_personality": env_personality
                })
                
                # All breakdowns
                personality_breakdown[user_personality][metric_name].append(score)
                environment_breakdown[env_personality][metric_name].append(score)
                permutation_length_breakdown[permutation_length][metric_name].append(score)
                combined_breakdown[user_personality][env_personality][metric_name].append(score)
        
        # Calculate averages
        total_conversations = len(conversations)
        if total_conversations > 0:
            for metric_name in metric_names:
                metric_scores[metric_name] /= total_conversations
        
        # Calculate breakdown averages
        for personality, metrics in personality_breakdown.items():
            for metric_name, scores in metrics.items():
                if scores:
                    personality_breakdown[personality][metric_name] = sum(scores) / len(scores)
                else:
                    personality_breakdown[personality][metric_name] = 0.0
        
        for env_type, metrics in environment_breakdown.items():
            for metric_name, scores in metrics.items():
                if scores:
                    environment_breakdown[env_type][metric_name] = sum(scores) / len(scores)
                else:
                    environment_breakdown[env_type][metric_name] = 0.0
        
        for length, metrics in permutation_length_breakdown.items():
            for metric_name, scores in metrics.items():
                if scores:
                    permutation_length_breakdown[length][metric_name] = sum(scores) / len(scores)
                else:
                    permutation_length_breakdown[length][metric_name] = 0.0
        
        for user_personality, env_dict in combined_breakdown.items():
            for env_personality, metrics in env_dict.items():
                for metric_name, scores in metrics.items():
                    if scores:
                        combined_breakdown[user_personality][env_personality][metric_name] = sum(scores) / len(scores)
                    else:
                        combined_breakdown[user_personality][env_personality][metric_name] = 0.0
        
        return EvaluationSummary(
            total_conversations=total_conversations,
            metric_scores=metric_scores,
            metric_details=metric_details,
            personality_breakdown=personality_breakdown,
            environment_breakdown=environment_breakdown,
            permutation_length_breakdown=permutation_length_breakdown,
            combined_breakdown=combined_breakdown
        )
    
    def print_summary(self, summary: EvaluationSummary) -> None:
        """Print a formatted summary of evaluation results."""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY ({summary.total_conversations} conversations)")
        print(f"{'='*60}")
        
        # Overall scores
        print("\nOVERALL SCORES:")
        for metric_name, score in summary.metric_scores.items():
            print(f"  {metric_name}: {score:.3f}")
        
        # Personality breakdown
        if summary.personality_breakdown:
            print("\nUSER PERSONALITY BREAKDOWN:")
            for personality, metrics in summary.personality_breakdown.items():
                print(f"  {personality}:")
                for metric_name, score in metrics.items():
                    print(f"    {metric_name}: {score:.3f}")
        
        # Environment breakdown
        if summary.environment_breakdown:
            print("\nENVIRONMENT BREAKDOWN:")
            for env_type, metrics in summary.environment_breakdown.items():
                print(f"  {env_type}:")
                for metric_name, score in metrics.items():
                    print(f"    {metric_name}: {score:.3f}")
        
        # Permutation length breakdown
        if summary.permutation_length_breakdown:
            print("\nPERMUTATION LENGTH BREAKDOWN:")
            for length, metrics in sorted(summary.permutation_length_breakdown.items()):
                print(f"  Length {length}:")
                for metric_name, score in metrics.items():
                    print(f"    {metric_name}: {score:.3f}")
        
        # Combined breakdown (if not too large)
        if summary.combined_breakdown and len(summary.combined_breakdown) <= 4:
            print("\nCOMBINED BREAKDOWN (User x Environment):")
            for user_personality, env_dict in summary.combined_breakdown.items():
                for env_personality, metrics in env_dict.items():
                    print(f"  {user_personality} x {env_personality}:")
                    for metric_name, score in metrics.items():
                        print(f"    {metric_name}: {score:.3f}")
        
        print(f"{'='*60}\n")
    
    def save_detailed_results(self, summary: EvaluationSummary, output_path: str) -> None:
        """Save detailed evaluation results to a JSON file."""
        import json
        
        # Convert to serializable format
        output_data = {
            "summary": asdict(summary),
            "timestamp": str(asyncio.get_event_loop().time()),
            "metrics_used": [metric.metric_name for metric in self.metrics]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Detailed evaluation results saved to {output_path}") 