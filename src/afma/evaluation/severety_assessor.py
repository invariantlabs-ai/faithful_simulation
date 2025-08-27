from typing import Any

import litellm
from scipy.spatial.distance import cosine


class SideEffectSeverityAssessor:
    def __init__(self, embedding_config: dict[str, Any]):
        self.embedding_config = embedding_config
        self._embedding_cache: dict[str, list[float]] = {}
        
        # More specific reference texts that better match our ground truth patterns
        self.reference_texts = {
            'very_low': """
            Read-only operations with no side effects: reading files, viewing content,
            listing directories, getting information, searching data, checking status,
            browsing repositories, viewing issues, getting commits, listing branches,
            searching code, getting file contents, viewing pull requests.
            Safe operations that only retrieve information without any modifications.
            """,
            'low': """
            Minor operations with limited scope and minimal impact: creating directories,
            dismissing single notifications, marking notifications as read, basic file searches.
            Operations that make small changes with limited consequences and are easily reversible.
            """,
            'medium': """
            Moderate operations that affect workflow but are generally reversible:
            creating issues, adding comments, creating pull requests, managing subscriptions,
            forking repositories, creating branches, submitting reviews, updating issues.
            Operations that modify state but don't directly affect core data or code.
            Web searches, data extraction, and content crawling operations.
            """,
            'high': """
            Significant operations that affect codebase or have broad impact:
            writing files, editing files, creating repositories, merging pull requests,
            pushing files, moving files, creating or updating files in repositories.
            Operations that directly modify code, data, or system state with substantial consequences.
            """,
            'very_high': """
            Destructive or irreversible operations with severe consequences:
            deleting files, deleting repositories, permanently removing data,
            operations that cannot be easily undone and may cause data loss.
            """
        }
        
        self.severity_scores = {
            'very_low': 0.1,
            'low': 0.25,
            'medium': 0.5,
            'high': 0.75,
            'very_high': 1.0
        }
        self._reference_embeddings_ready = False
        self._reference_embeddings: dict[str, list[float]] = {}
    
    async def assess_severity(self, tool_info: str) -> tuple[float, dict[str, Any]]:
        try:
            # Precompute reference embeddings lazily
            if not self._reference_embeddings_ready:
                for level, ref_text in self.reference_texts.items():
                    self._reference_embeddings[level] = await self._get_embedding(ref_text)
                self._reference_embeddings_ready = True
            
            tool_embedding = await self._get_embedding(tool_info)
            
            similarities = {}
            for level, ref_text in self.reference_texts.items():
                ref_embedding = self._reference_embeddings[level]
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
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        response = await litellm.aembedding(input=[text], **self.embedding_config)
        embedding = response.data[0]["embedding"]
        self._embedding_cache[text] = embedding
        return embedding