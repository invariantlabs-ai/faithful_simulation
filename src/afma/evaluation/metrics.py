from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import litellm
from loguru import logger
import hashlib
import json
import math
from scipy.special import softmax
from scipy.spatial.distance import cosine


@dataclass
class EvaluationResult:
    """Result of an evaluation metric."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    async def evaluate(self, 
                      user_goal: str, 
                      user_source: List[Dict[str, str]], 
                      execution_trace: List[Dict[str, Any]], 
                      used_tools: List[str]) -> EvaluationResult:
        """
        Evaluate a single conversation trace.
        
        Args:
            user_goal: The original user goal/request
            user_source: The expected tool sequence from user generation
            execution_trace: Full conversation history including tool calls
            used_tools: List of tool names that were actually used
            
        Returns:
            EvaluationResult with score and details
        """
        pass


class WeightedLevenshteinMetric(EvaluationMetric):
    """
    Weighted Levenshtein Distance metric that uses semantic similarity between tools.
    
    Uses embeddings to calculate semantic similarity between tools and applies
    weighted costs in the Levenshtein algorithm based on semantic distance.
    """
    
    def __init__(self, 
                 tool_definitions: Dict[str, Dict[str, Any]],
                 embedding_config: Dict[str, Any],
                 similarity_threshold: float = 0.8,
                 cache_embeddings: bool = True):
        """
        Initialize the weighted Levenshtein metric.
        
        Args:
            tool_definitions: Dict mapping tool names to their definitions
            embedding_config: Configuration for litellm embedding calls
            similarity_threshold: Threshold above which tools are considered similar
            cache_embeddings: Whether to cache embeddings for efficiency
        """
        self.metric_name = "weighted_levenshtein"
        self.tool_definitions = tool_definitions
        self.embedding_config = embedding_config
        self.similarity_threshold = similarity_threshold
        self.cache_embeddings = cache_embeddings
        
        # Cache for embeddings and similarity scores
        self._embedding_cache: Dict[str, List[float]] = {}
        self._similarity_cache: Dict[tuple, float] = {}
    
    async def evaluate(self, 
                      user_goal: str, 
                      user_source: List[Dict[str, str]], 
                      execution_trace: List[Dict[str, Any]], 
                      used_tools: List[str]) -> EvaluationResult:
        """
        Calculate weighted Levenshtein distance between expected and used tool sequences.
        """
        try:
            # Extract expected tool sequence from user_source
            expected_tools = [tool["name"] for tool in user_source]
            
            # Calculate weighted Levenshtein distance using optimized alignment
            alignment_result = await self.get_optimal_alignment(expected_tools, used_tools)
            distance = alignment_result["distance"]
            
            # Normalize to similarity score (0-1 range)
            max_len = max(len(expected_tools), len(used_tools)) if expected_tools or used_tools else 1
            similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
            
            return EvaluationResult(
                metric_name=self.metric_name,
                score=similarity,
                details={
                    "expected_tools": expected_tools,
                    "used_tools": used_tools,
                    "raw_distance": distance,
                    "max_length": max_len,
                    "expected_length": len(expected_tools),
                    "used_length": len(used_tools),
                    "alignment": alignment_result["alignment"],
                    "operations": alignment_result["operations"]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in weighted Levenshtein evaluation: {e}")
            return EvaluationResult(
                metric_name=self.metric_name,
                score=0.0,
                error=str(e)
            )
    
    async def _weighted_levenshtein_distance(self, seq1: List[str], seq2: List[str]) -> float:
        """
        Calculate weighted Levenshtein distance using semantic similarity.
        This is now a lightweight wrapper around get_optimal_alignment.
        """
        alignment_result = await self.get_optimal_alignment(seq1, seq2)
        return alignment_result["distance"]
    
    async def _get_tool_similarity(self, tool1: str, tool2: str) -> float:
        """
        Get semantic similarity between two tools using normalized embeddings across all tools.
        Compares tool1 against all possible tools, normalizes similarities, and returns 
        the normalized similarity for tool2.
        """
        cache_key = tuple(sorted([tool1, tool2]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        try:
            query_embedding = await self._get_tool_embedding(tool1)
            
            all_tools = list(self.tool_definitions.keys())
            all_similarities = []
            
            for tool in all_tools:
                tool_embedding = await self._get_tool_embedding(tool)
                similarity = 1 - cosine(query_embedding, tool_embedding)
                all_similarities.append(similarity)
            
            normalized_similarities = softmax(all_similarities).tolist()
            
            tool2_index = all_tools.index(tool2)
            similarity = normalized_similarities[tool2_index]
            
            if self.cache_embeddings:
                self._similarity_cache[cache_key] = similarity
            
            return similarity
            
        except Exception as e:
            logger.exception(f"Error calculating similarity between {tool1} and {tool2}: {e}")
            return 0.0
    
    async def _get_tool_embedding(self, tool_name: str) -> List[float]:
        """
        Get embedding for a tool, using cache if available.
        """
        if self.cache_embeddings and tool_name in self._embedding_cache:
            return self._embedding_cache[tool_name]
        
        tool_text = self._format_tool_for_embedding(tool_name)
        
        response = await litellm.aembedding(
            input=[tool_text],
            **self.embedding_config
        )
        
        embedding = response.data[0]["embedding"]
        
        if self.cache_embeddings:
            self._embedding_cache[tool_name] = embedding
        
        return embedding
    
    def _format_tool_for_embedding(self, tool_name: str) -> str:
        """
        Format tool information for embedding generation.
        Similar to _format_tool_information but for a single tool.
        """
        if tool_name not in self.tool_definitions:
            logger.warning(f"Tool {tool_name} not found in definitions, using name only")
            return tool_name
        
        tool = self.tool_definitions[tool_name]
        
        formatted = f"{tool_name}: {tool.get('description', 'No description available')}"
        
        if "inputSchema" in tool and "required" in tool["inputSchema"] and tool["inputSchema"]["required"]:
            required_params = tool["inputSchema"]["required"]
            formatted += "\nFunction parameters:"
            properties = tool["inputSchema"]["properties"]
            
            for param_name in required_params:
                param_description = properties[param_name].get("description", None)
                if param_description is None:
                    formatted += f"\n- {param_name}"
                else:
                    formatted += f"\n- {param_name}: {param_description}"
        
        return formatted

    async def get_optimal_alignment(self, seq1: List[str], seq2: List[str]) -> Dict[str, Any]:
        """
        Get the optimal alignment between two tool sequences.
        
        Returns:
            Dictionary containing:
            - distance: The weighted Levenshtein distance
            - alignment: List of tuples representing the alignment
            - operations: List of operations (match, substitute, insert, delete)
        """
        if not seq1 and not seq2:
            return {
                "distance": 0.0,
                "alignment": [],
                "operations": []
            }
        
        if not seq1 or not seq2:
            # Handle empty sequences
            if not seq1:
                operations = [("insert", None, tool) for tool in seq2]
                alignment = [(None, tool) for tool in seq2]
            else:
                operations = [("delete", tool, None) for tool in seq1]
                alignment = [(tool, None) for tool in seq1]
            
            return {
                "distance": float(max(len(seq1), len(seq2))),
                "alignment": alignment,
                "operations": operations
            }
        
        len1, len2 = len(seq1), len(seq2)
        
        # Create matrices to store distances and operation choices
        dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]
        operations_matrix = [[None] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = float(i)
            if i > 0:
                operations_matrix[i][0] = "delete"
        for j in range(len2 + 1):
            dp[0][j] = float(j)
            if j > 0:
                operations_matrix[0][j] = "insert"
        
        # Fill the matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                    operations_matrix[i][j] = "match"
                else:
                    # Calculate semantic similarity for substitution cost
                    similarity = await self._get_tool_similarity(seq1[i-1], seq2[j-1])
                    substitution_cost = 1.0 - similarity
                    
                    deletion_cost = dp[i-1][j] + 1.0
                    insertion_cost = dp[i][j-1] + 1.0
                    substitution_total_cost = dp[i-1][j-1] + substitution_cost
                    
                    min_cost = min(deletion_cost, insertion_cost, substitution_total_cost)
                    
                    if min_cost == deletion_cost:
                        dp[i][j] = deletion_cost
                        operations_matrix[i][j] = "delete"
                    elif min_cost == insertion_cost:
                        dp[i][j] = insertion_cost
                        operations_matrix[i][j] = "insert"
                    else:
                        dp[i][j] = substitution_total_cost
                        operations_matrix[i][j] = "substitute"
        
        # Backtrack to get the alignment
        alignment = []
        operations = []
        i, j = len1, len2
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                operation = operations_matrix[i][j]
                if operation == "match":
                    alignment.append((seq1[i-1], seq2[j-1]))
                    operations.append(("match", seq1[i-1], seq2[j-1]))
                    i -= 1
                    j -= 1
                elif operation == "substitute":
                    alignment.append((seq1[i-1], seq2[j-1]))
                    operations.append(("substitute", seq1[i-1], seq2[j-1]))
                    i -= 1
                    j -= 1
                elif operation == "delete":
                    alignment.append((seq1[i-1], None))
                    operations.append(("delete", seq1[i-1], None))
                    i -= 1
                elif operation == "insert":
                    alignment.append((None, seq2[j-1]))
                    operations.append(("insert", None, seq2[j-1]))
                    j -= 1
            elif i > 0:
                alignment.append((seq1[i-1], None))
                operations.append(("delete", seq1[i-1], None))
                i -= 1
            else:
                alignment.append((None, seq2[j-1]))
                operations.append(("insert", None, seq2[j-1]))
                j -= 1
        
        # Reverse to get correct order
        alignment.reverse()
        operations.reverse()
        
        return {
            "distance": dp[len1][len2],
            "alignment": alignment,
            "operations": operations
        }

    async def align_multiple_sequences(self, reference_sequence: List[str], sequences: List[List[str]]) -> Dict[str, Any]:
        """
        Align multiple tool sequences against a reference (ground truth) sequence.
        
        Args:
            reference_sequence: The expected/ground truth tool sequence
            sequences: List of actual tool sequences to align against the reference
            
        Returns:
            Dictionary containing alignment information for all sequences
        """
        if not sequences:
            return {"alignments": [], "reference_sequence": reference_sequence}
        
        if not reference_sequence:
            # If no reference sequence, treat first sequence as reference for compatibility
            reference_sequence = sequences[0] if sequences else []
        
        alignments = []
        
        for seq in sequences:
            alignment_info = await self.get_optimal_alignment(reference_sequence, seq)
            alignments.append(alignment_info)
        
        return {
            "alignments": alignments,
            "reference_sequence": reference_sequence
        } 