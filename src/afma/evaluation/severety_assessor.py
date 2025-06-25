from typing import Any

import litellm
from scipy.spatial.distance import cosine


class SideEffectSeverityAssessor:
    def __init__(self, embedding_config: dict[str, Any]):
        self.embedding_config = embedding_config
        
        self.reference_texts = {
            'very_low': """
            Completely safe read-only operations: viewing files, reading data,
            getting information, checking status, validating input, calculating values.
            """,
            'low': """
            Safe operations with minimal impact: searching, analyzing, formatting,
            converting data, parsing content, generating reports.
            """,
            'medium': """
            Moderate impact operations: saving files, logging data, caching information,
            downloading content, organizing files, creating backups.
            """,
            'high': """
            Significant impact operations: modifying files, uploading data,
            sending notifications, connecting to services, updating databases.
            """,
            'very_high': """
            Dangerous operations with severe consequences: deleting data,
            sending money, executing code, publishing content, terminating processes.
            """
        }
        
        self.severity_scores = {
            'very_low': 0.0,
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75,
            'very_high': 1.0
        }
    
    async def assess_severity(self, tool_info: str) -> tuple[float, dict[str, Any]]:
        try:
            tool_embedding = await self._get_embedding(tool_info)
            
            similarities = {}
            for level, ref_text in self.reference_texts.items():
                ref_embedding = await self._get_embedding(ref_text)
                similarity = 1 - cosine(tool_embedding, ref_embedding)
                similarities[level] = similarity
            
            best_match = max(similarities.keys(), key=lambda k: similarities[k])
            score = self.severity_scores[best_match]
            
            return score, {
                'similarities': similarities,
                'best_match': best_match,
                'confidence': similarities[best_match]
            }
            
        except Exception as e:
            return 0.5, {'error': str(e)}
    
    async def _get_embedding(self, text: str) -> list[float]:
        response = await litellm.aembedding(input=[text], **self.embedding_config)
        embedding = response.data[0]["embedding"]
        return embedding