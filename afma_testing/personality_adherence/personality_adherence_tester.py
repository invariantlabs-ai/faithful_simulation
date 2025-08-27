import asyncio
import json
import os
import statistics
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import glob

import matplotlib.pyplot as plt
import numpy as np
import colorsys
from loguru import logger
import yaml

from .personality_adherence_assessor import PersonalityAdherenceAssessor


@dataclass
class PersonalityAdherenceTestResult:
    """Result of personality adherence testing for a single conversation."""
    conversation_id: int
    user_personality: Optional[str]
    user_personality_name: Optional[str]
    environment_personality: Optional[str]
    environment_personality_name: Optional[str]
    trace_set_id: Optional[str]
    instantiation_id: Optional[int]
    
    # Assessment results
    user_personality_adherence_score: float
    environment_personality_adherence_score: float
    
    # Data for debugging
    user_messages: str
    tool_interactions: str

    # Metadata
    assessment_timestamp: str


@dataclass
class PersonalityAdherenceTestSummary:
    """Summary of all personality adherence test results."""
    total_conversations: int
    
    # User personality metrics
    avg_user_personality_score: float
    std_user_personality_score: float
    
    # Environment personality metrics
    avg_environment_personality_score: float
    std_environment_personality_score: float

    # Results by personality
    results_by_user_personality: Dict[str, Dict[str, float]]
    results_by_environment_personality: Dict[str, Dict[str, float]]
    
    # Detailed results
    results: List[PersonalityAdherenceTestResult]
    
    # Configuration
    config: Dict[str, Any]


class PersonalityAdherenceTester:
    """Main tester for personality adherence evaluation."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.assessor = PersonalityAdherenceAssessor(
            llm_config=self.config["personality_adherence_testing"]["assessment_litellm"],
        )
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_fields = [
            "conversations_path",
            "personality_adherence_testing.assessment_litellm",
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
        conversations_path = self.config["conversations_path"]
        if not os.path.exists(conversations_path):
            raise FileNotFoundError(f"Conversations file not found: {conversations_path}")
        
        logger.info(f"Loading conversations from {conversations_path}")
        with open(conversations_path, 'r') as f:
            conversations = json.load(f)
        
        logger.success(f"Loaded {len(conversations)} conversations")
        return conversations
    
    def _load_existing_results(self) -> Optional[Tuple[List[PersonalityAdherenceTestResult], PersonalityAdherenceTestSummary]]:
        """Load existing results from output directory if available."""
        output_dir = self.config.get("output_dir", "results/personality_adherence")
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return None
            
        # Look for the most recent results and summary files
        results_files = list(output_path.glob("results_*.json"))
        summary_files = list(output_path.glob("summary_*.json"))
        
        if not results_files or not summary_files:
            return None
            
        # Get the most recent files
        latest_results_file = max(results_files, key=lambda p: p.stat().st_mtime)
        latest_summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
        
        try:
            logger.info(f"Loading existing results from {latest_results_file}")
            with open(latest_results_file, 'r') as f:
                results_data = json.load(f)
            
            logger.info(f"Loading existing summary from {latest_summary_file}")
            with open(latest_summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Convert back to dataclasses
            results = [PersonalityAdherenceTestResult(**result) for result in results_data]
            
            # Reconstruct summary (results field will be updated)
            summary_data['results'] = results
            summary = PersonalityAdherenceTestSummary(**summary_data)
            
            logger.success(f"Loaded {len(results)} existing test results")
            return results, summary
            
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return None

    async def run_test(self) -> PersonalityAdherenceTestSummary:
        logger.info("Starting personality adherence testing")
        
        # Try to load existing results first
        existing_data = self._load_existing_results()
        if existing_data:
            test_results, summary = existing_data
            logger.info("Using existing test results - skipping test execution")
        else:
            logger.info("No existing results found - running new tests")
            conversations = self._load_conversations()
            
            tasks = [self.assessor.assess_conversation(conv) for conv in conversations]
            assessment_results = await asyncio.gather(*tasks)
            
            test_results = []
            for i, (conversation, (user_score, env_score, user_messages, tool_interactions)) in enumerate(zip(conversations, assessment_results)):
                result = self._process_conversation_results(
                    i, conversation, user_score, env_score, user_messages, tool_interactions
                )
                test_results.append(result)
                
            summary = self._create_summary(test_results)
            await self._save_results(test_results, summary)
        
        # Generate visualizations
        self._create_visualizations(summary)
        
        logger.success("Personality adherence testing completed")
        return summary

    def _process_conversation_results(self, conversation_id: int, conversation: Dict[str, Any], 
                                      user_score: float, env_score: float,
                                      user_messages: str, tool_interactions: str) -> PersonalityAdherenceTestResult:
        return PersonalityAdherenceTestResult(
            conversation_id=conversation_id,
            user_personality=conversation.get("user_personality"),
            user_personality_name=conversation.get("user_personality_name"),
            environment_personality=conversation.get("environment_personality"),
            environment_personality_name=conversation.get("environment_personality_name"),
            trace_set_id=conversation.get("trace_set_id"),
            instantiation_id=conversation.get("instantiation_id"),
            user_personality_adherence_score=user_score,
            environment_personality_adherence_score=env_score,
            user_messages=user_messages,
            tool_interactions=tool_interactions,
            assessment_timestamp=datetime.now().isoformat()
        )

    def _create_summary(self, results: List[PersonalityAdherenceTestResult]) -> PersonalityAdherenceTestSummary:
        total_conversations = len(results)
        
        user_scores = [r.user_personality_adherence_score for r in results]
        env_scores = [r.environment_personality_adherence_score for r in results]
        
        avg_user_score = statistics.mean(user_scores) if user_scores else 0.0
        std_user_score = statistics.stdev(user_scores) if len(user_scores) > 1 else 0.0
        
        avg_env_score = statistics.mean(env_scores) if env_scores else 0.0
        std_env_score = statistics.stdev(env_scores) if len(env_scores) > 1 else 0.0
        
        # Results by personality
        results_by_user_personality = self._aggregate_by_personality(
            results, "user_personality_name", "user_personality_adherence_score"
        )
        results_by_environment_personality = self._aggregate_by_personality(
            results, "environment_personality_name", "environment_personality_adherence_score"
        )

        return PersonalityAdherenceTestSummary(
            total_conversations=total_conversations,
            avg_user_personality_score=avg_user_score,
            std_user_personality_score=std_user_score,
            avg_environment_personality_score=avg_env_score,
            std_environment_personality_score=std_env_score,
            results_by_user_personality=results_by_user_personality,
            results_by_environment_personality=results_by_environment_personality,
            results=results,
            config=self.config
        )

    def _aggregate_by_personality(self, results: List[PersonalityAdherenceTestResult],
                                personality_field: str, score_field: str) -> Dict[str, Dict[str, float]]:
        """Aggregate results by a specific personality field."""
        by_personality: Dict[str, Dict[str, Any]] = {}
        for result in results:
            name = getattr(result, personality_field, "Unknown")
            if name is None:
                name = "Unknown"

            if name not in by_personality:
                by_personality[name] = {"scores": [], "count": 0}

            by_personality[name]["count"] += 1
            score = getattr(result, score_field)
            by_personality[name]["scores"].append(score)

        # Calculate final metrics
        summary_by_personality: Dict[str, Dict[str, float]] = {}
        for name, data in by_personality.items():
            avg_score = statistics.mean(data["scores"]) if data["scores"] else 0.0
            std_dev = statistics.stdev(data["scores"]) if len(data["scores"]) > 1 else 0.0
            summary_by_personality[name] = {
                "count": data["count"],
                "avg_score": avg_score,
                "std_dev": std_dev
            }
        return summary_by_personality

    async def _save_results(self, results: List[PersonalityAdherenceTestResult], 
                          summary: PersonalityAdherenceTestSummary):
        output_dir = self.config.get("output_dir", "results/personality_adherence")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_path = Path(output_dir) / f"results_{timestamp}.json"
        summary_path = Path(output_dir) / f"summary_{timestamp}.json"
        
        logger.info(f"Saving detailed results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
            
        logger.info(f"Saving summary to {summary_path}")
        with open(summary_path, 'w') as f:
            # A bit of a hack to handle the dataclass-in-dataclass for json serialization
            summary_dict = asdict(summary)
            summary_dict["results"] = [asdict(r) for r in summary.results]
            json.dump(summary_dict, f, indent=2)

    def _create_visualizations(self, summary: PersonalityAdherenceTestSummary):
        """Create visualization graphs for personality adherence results."""
        output_dir = self.config.get("output_dir", "results/personality_adherence")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating personality adherence visualizations")
        
        # Create circular radar chart
        self._create_radar_chart(summary, output_dir)
        
        # Create LaTeX table
        self._create_latex_table(summary, output_dir)
        
        logger.success("Visualizations created successfully")

    def _create_radar_chart(self, summary: PersonalityAdherenceTestSummary, output_dir: str):
        """Create single circular radar chart with both user and environment personality adherence scores."""
        # Get user and environment personalities
        user_personalities = list(summary.results_by_user_personality.keys()) if summary.results_by_user_personality else []
        env_personalities = list(summary.results_by_environment_personality.keys()) if summary.results_by_environment_personality else []
        
        # Combine all personalities for the full circle
        all_personalities = user_personalities + env_personalities
        n_personalities = len(all_personalities)
        
        if n_personalities == 0:
            logger.warning("No personalities found to plot")
            return
        
        # Calculate angles: left semicircle for users, right semicircle for environments
        # Left semicircle: π/2 to 3π/2 (90° to 270°, top going counterclockwise to bottom)
        if len(user_personalities) > 0:
            user_angles = np.linspace(np.pi/2, 3*np.pi/2, len(user_personalities) + 1, endpoint=False).tolist()[1:]
        else:
            user_angles = []
        
        # Right semicircle: 3π/2 to π/2 (270° to 90°, bottom going counterclockwise to top)
        if len(env_personalities) > 0:
            # Create angles from 3π/2 to 5π/2 then convert to [0, 2π] range
            raw_env_angles = np.linspace(3*np.pi/2, 5*np.pi/2, len(env_personalities) + 1, endpoint=False)[1:]
            env_angles = [(angle % (2*np.pi)) for angle in raw_env_angles]
        else:
            env_angles = []
        
        angles = user_angles + env_angles
        
        # Create single figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Set title
        ax.set_title("Archetype Adherence Scores", pad=20, fontsize=16, fontweight='bold')
        
        # Prepare data for all personalities
        all_scores = []
        all_angles = []
        
        # Add user personality scores (left semicircle)
        for i, personality in enumerate(user_personalities):
            if personality in summary.results_by_user_personality:
                all_scores.append(summary.results_by_user_personality[personality]['avg_score'])
                all_angles.append(user_angles[i])
        
        # Add environment personality scores (right semicircle)
        for i, personality in enumerate(env_personalities):
            if personality in summary.results_by_environment_personality:
                all_scores.append(summary.results_by_environment_personality[personality]['avg_score'])
                all_angles.append(env_angles[i])
        
        if all_scores:
            # Close the polygon
            all_scores += [all_scores[0]]
            all_angles += [all_angles[0]]
            
            # Plot the data
            ax.plot(all_angles, all_scores, 'o-', linewidth=3, 
                   label='Archetype Adherence', color='steelblue', alpha=0.8, markersize=8)
            ax.fill(all_angles, all_scores, alpha=0.15, color='steelblue')
        
        # Configure chart
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels(all_personalities, fontsize=10)
        ax.grid(True, color='darkgrey', alpha=0.7, linewidth=1.0)
        ax.set_facecolor('white')
        
        # Add vertical divider line and labels to separate user and environment sides
        ax.plot([np.pi/2, np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Top vertical line
        ax.plot([3*np.pi/2, 3*np.pi/2], [0, 1], 'k-', linewidth=2, alpha=0.7)  # Bottom vertical line
        ax.text(np.pi, 0.5, 'User', fontsize=14, fontweight='bold', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax.text(0, 0.5, 'Environment', fontsize=14, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = Path(output_dir) / "personality_adherence_radar_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Radar chart saved to {chart_path}")

    def _create_latex_table(self, summary: PersonalityAdherenceTestSummary, output_dir: str):
        """Create LaTeX table with personality adherence results."""
        latex_content = []
        
        # Table header
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Personality Adherence Test Results}")
        latex_content.append("\\label{tab:personality_adherence}")
        latex_content.append("\\begin{tabular}{lccc}")
        latex_content.append("\\toprule")
        latex_content.append("Personality Type & Conversations & Avg Score & Std Dev \\\\")
        latex_content.append("\\midrule")
        
        # User personalities section
        if summary.results_by_user_personality:
            latex_content.append("\\multicolumn{4}{c}{\\textbf{User Personalities}} \\\\")
            latex_content.append("\\midrule")
            
            for personality, data in summary.results_by_user_personality.items():
                latex_content.append(f"{personality} & {data['count']} & {data['avg_score']:.2f} & {data['std_dev']:.2f} \\\\")
        
        # Environment personalities section
        if summary.results_by_environment_personality:
            latex_content.append("\\midrule")
            latex_content.append("\\multicolumn{4}{c}{\\textbf{Environment Personalities}} \\\\")
            latex_content.append("\\midrule")
            
            for personality, data in summary.results_by_environment_personality.items():
                latex_content.append(f"{personality} & {data['count']} & {data['avg_score']:.2f} & {data['std_dev']:.2f} \\\\")
        
        # Table footer
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        # Save LaTeX table
        latex_path = Path(output_dir) / "personality_adherence_table.tex"
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_content))
        
        logger.info(f"LaTeX table saved to {latex_path}") 