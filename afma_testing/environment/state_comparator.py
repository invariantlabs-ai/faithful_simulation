"""
State comparator that compares real and simulated environment states
and generates similarity scores using text-based difference calculation.
"""

import json
from typing import Dict, Any, Tuple
from loguru import logger
from difflib import SequenceMatcher


class StateComparator:
    """Compares environment states and generates similarity scores."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.comparison_config = config.get("comparison", {})
    
    async def compare_states(self, real_state: Dict[str, str], 
                           simulated_state: Dict[str, str],
                           task_description: str = "") -> Dict[str, Any]:
        """
        Compare real and simulated environment states.
        
        Args:
            real_state: State from real environment
            simulated_state: State from simulated environment
            task_description: Description of the task that was performed
            
        Returns:
            Dictionary with comparison results including similarity score
        """
        try:
            logger.info("Comparing real and simulated environment states")
            
            # Calculate similarity score using text-based comparison
            similarity_score = self._calculate_similarity_score(real_state, simulated_state)
            
            # Generate detailed comparison analysis
            detailed_comparison = self._generate_detailed_comparison(real_state, simulated_state)
            
            # Create comprehensive result
            result = {
                "similarity_score": similarity_score,
                "detailed_comparison": detailed_comparison,
                "real_state_file_count": len(real_state),
                "simulated_state_file_count": len(simulated_state),
                "real_files": list(real_state.keys()),
                "simulated_files": list(simulated_state.keys()),
                "missing_in_simulated": list(set(real_state.keys()) - set(simulated_state.keys())),
                "extra_in_simulated": list(set(simulated_state.keys()) - set(real_state.keys())),
                "common_files": list(set(real_state.keys()) & set(simulated_state.keys()))
            }
            
            logger.success(f"Comparison completed. Similarity score: {similarity_score:.3f}")
            return result
            
        except Exception as e:
            logger.exception(f"Error comparing states: {e}")
            return {
                "similarity_score": 0.0,
                "error": str(e),
                "real_state_file_count": len(real_state),
                "simulated_state_file_count": len(simulated_state)
            }
    
    def _calculate_similarity_score(self, real_state: Dict[str, str], 
                                  simulated_state: Dict[str, str]) -> float:
        """
        Calculate similarity score between two states using text-based comparison.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not real_state and not simulated_state:
            return 1.0  # Both empty states are identical
        
        if not real_state or not simulated_state:
            return 0.0  # One empty, one not - completely different
        
        # Get all unique file paths
        all_files = set(real_state.keys()) | set(simulated_state.keys())
        real_files = set(real_state.keys())
        simulated_files = set(simulated_state.keys())
        
        # Calculate structure similarity (file presence/absence)
        common_files = real_files & simulated_files
        missing_files = real_files - simulated_files
        extra_files = simulated_files - real_files
        
        total_files = len(all_files)
        structure_score = len(common_files) / total_files if total_files > 0 else 1.0
        
        # Calculate content similarity for common files (exact path matches)
        exact_content_scores = []
        for file_path in common_files:
            real_content = real_state[file_path]
            simulated_content = simulated_state[file_path]
            
            if real_content == simulated_content:
                exact_content_scores.append(1.0)
            else:
                # Use SequenceMatcher for fuzzy content comparison
                similarity = SequenceMatcher(None, real_content, simulated_content).ratio()
                exact_content_scores.append(similarity)
        
        # Calculate content similarity for files with different paths but similar content
        cross_content_scores = []
        unmatched_real = real_files - common_files
        unmatched_simulated = simulated_files - common_files
        
        # For each unmatched file in real state, find best content match in simulated state
        for real_path in unmatched_real:
            real_content = real_state[real_path]
            best_similarity = 0.0
            
            for sim_path in unmatched_simulated:
                sim_content = simulated_state[sim_path]
                if real_content == sim_content:
                    best_similarity = 1.0
                    break
                else:
                    similarity = SequenceMatcher(None, real_content, sim_content).ratio()
                    best_similarity = max(best_similarity, similarity)
            
            if best_similarity > 0.8:  # Only count as significant match if similarity > 80%
                cross_content_scores.append(best_similarity * 0.8)  # Penalize for path difference
        
        # Combine exact and cross-content scores
        all_content_scores = exact_content_scores + cross_content_scores
        content_score = sum(all_content_scores) / len(all_content_scores) if all_content_scores else 0.0
        
        # Weighted combination: structure (30%) + content (70%)
        # Give more weight to content since we now handle path differences
        final_score = (0.3 * structure_score) + (0.7 * content_score)
        
        return final_score
    
    def _generate_detailed_comparison(self, real_state: Dict[str, str], 
                                    simulated_state: Dict[str, str]) -> str:
        """Generate detailed comparison analysis without using LLM."""
        
        real_files = set(real_state.keys())
        simulated_files = set(simulated_state.keys())
        common_files = real_files & simulated_files
        missing_files = real_files - simulated_files
        extra_files = simulated_files - real_files
        
        analysis = []
        analysis.append("DETAILED COMPARISON ANALYSIS")
        analysis.append("=" * 50)
        
        # File structure analysis
        analysis.append(f"\nFILE STRUCTURE ANALYSIS:")
        analysis.append(f"- Total files in real state: {len(real_state)}")
        analysis.append(f"- Total files in simulated state: {len(simulated_state)}")
        analysis.append(f"- Common files (exact path match): {len(common_files)}")
        analysis.append(f"- Missing in simulation: {len(missing_files)}")
        analysis.append(f"- Extra in simulation: {len(extra_files)}")
        
        if missing_files:
            analysis.append(f"\nMISSING FILES IN SIMULATION:")
            for file_path in sorted(missing_files):
                analysis.append(f"  - {file_path}")
        
        if extra_files:
            analysis.append(f"\nEXTRA FILES IN SIMULATION:")
            for file_path in sorted(extra_files):
                analysis.append(f"  - {file_path}")
        
        # Content analysis for common files (exact path matches)
        if common_files:
            analysis.append(f"\nCONTENT ANALYSIS FOR EXACT PATH MATCHES:")
            exact_content_differences = []
            
            for file_path in sorted(common_files):
                real_content = real_state[file_path]
                simulated_content = simulated_state[file_path]
                
                if real_content == simulated_content:
                    analysis.append(f"  ✓ {file_path}: Identical content")
                else:
                    similarity = SequenceMatcher(None, real_content, simulated_content).ratio()
                    analysis.append(f"  ⚠ {file_path}: {similarity:.3f} similarity")
                    exact_content_differences.append((file_path, similarity))
            
            if exact_content_differences:
                analysis.append(f"\nEXACT PATH CONTENT DIFFERENCES:")
                for file_path, similarity in exact_content_differences:
                    analysis.append(f"  - {file_path}: {similarity:.3f} similarity")
        
        # Cross-path content analysis
        unmatched_real = real_files - common_files
        unmatched_simulated = simulated_files - common_files
        
        if unmatched_real and unmatched_simulated:
            analysis.append(f"\nCROSS-PATH CONTENT ANALYSIS:")
            cross_matches = []
            
            for real_path in sorted(unmatched_real):
                real_content = real_state[real_path]
                best_match = None
                best_similarity = 0.0
                
                for sim_path in unmatched_simulated:
                    sim_content = simulated_state[sim_path]
                    if real_content == sim_content:
                        best_match = sim_path
                        best_similarity = 1.0
                        break
                    else:
                        similarity = SequenceMatcher(None, real_content, sim_content).ratio()
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = sim_path
                
                if best_similarity > 0.8:  # Only show significant matches
                    analysis.append(f"  🔄 {real_path} ↔ {best_match}: {best_similarity:.3f} similarity")
                    cross_matches.append((real_path, best_match, best_similarity))
            
            if cross_matches:
                analysis.append(f"\nCROSS-PATH MATCHES SUMMARY:")
                for real_path, sim_path, similarity in cross_matches:
                    analysis.append(f"  - {real_path} ↔ {sim_path}: {similarity:.3f} similarity")
            else:
                analysis.append("  No significant cross-path content matches found.")
        
        # Calculate and display scores using the same logic as _calculate_similarity_score
        structure_score = len(common_files) / len(real_files | simulated_files) if (real_files | simulated_files) else 1.0
        
        # Calculate exact content scores
        exact_content_scores = []
        for file_path in common_files:
            real_content = real_state[file_path]
            simulated_content = simulated_state[file_path]
            if real_content == simulated_content:
                exact_content_scores.append(1.0)
            else:
                similarity = SequenceMatcher(None, real_content, simulated_content).ratio()
                exact_content_scores.append(similarity)
        
        # Calculate cross-content scores
        cross_content_scores = []
        for real_path in unmatched_real:
            real_content = real_state[real_path]
            best_similarity = 0.0
            
            for sim_path in unmatched_simulated:
                sim_content = simulated_state[sim_path]
                if real_content == sim_content:
                    best_similarity = 1.0
                    break
                else:
                    similarity = SequenceMatcher(None, real_content, sim_content).ratio()
                    best_similarity = max(best_similarity, similarity)
            
            if best_similarity > 0.8:
                cross_content_scores.append(best_similarity * 0.8)
        
        # Combine scores
        all_content_scores = exact_content_scores + cross_content_scores
        content_score = sum(all_content_scores) / len(all_content_scores) if all_content_scores else 0.0
        final_score = (0.3 * structure_score) + (0.7 * content_score)
        
        analysis.append(f"\nSCORING BREAKDOWN:")
        analysis.append(f"- Structure similarity: {structure_score:.3f}")
        analysis.append(f"- Content similarity: {content_score:.3f}")
        analysis.append(f"- Final weighted score: {final_score:.3f}")
        
        if cross_content_scores:
            analysis.append(f"- Cross-path matches found: {len(cross_content_scores)}")
            analysis.append(f"- Cross-path average: {sum(cross_content_scores) / len(cross_content_scores):.3f}")
        
        return "\n".join(analysis)
    
    def get_comparison_summary(self, comparison_results: list) -> Dict[str, Any]:
        """Generate a summary of multiple comparison results."""
        if not comparison_results:
            return {"error": "No comparison results provided"}
        
        scores = [result.get("similarity_score", 0.0) for result in comparison_results]
        
        summary = {
            "total_tests": len(comparison_results),
            "average_similarity": sum(scores) / len(scores) if scores else 0.0,
            "min_similarity": min(scores) if scores else 0.0,
            "max_similarity": max(scores) if scores else 0.0,
            "scores_distribution": {
                "excellent": len([s for s in scores if s >= 0.9]),
                "good": len([s for s in scores if 0.7 <= s < 0.9]),
                "fair": len([s for s in scores if 0.5 <= s < 0.7]),
                "poor": len([s for s in scores if s < 0.5])
            }
        }
        
        return summary 