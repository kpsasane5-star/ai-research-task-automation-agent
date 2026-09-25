import math
from typing import List, Dict, Any
from src.rag.vector_store import VectorStoreManager
from config import TOP_K_RETRIEVAL
from src.utils.logger import logger

class HybridRetriever:
    """Hybrid Retriever combining Dense Vector Search with BM25 Keyword Search."""

    def __init__(self, vector_store: VectorStoreManager, top_k: int = TOP_K_RETRIEVAL):
        self.vector_store = vector_store
        self.top_k = top_k

    def _bm25_score(self, query: str, document: str) -> float:
        """Compute lightweight BM25-style keyword matching score."""
        query_words = set(query.lower().split())
        doc_words = document.lower().split()
        if not doc_words or not query_words:
            return 0.0

        doc_len = len(doc_words)
        score = 0.0
        k1 = 1.5
        b = 0.75
        avg_len = 200.0  # approximate average chunk length

        for q in query_words:
            tf = doc_words.count(q)
            if tf > 0:
                idf = math.log(1 + 10.0 / (1 + tf))
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * (doc_len / avg_len))
                score += idf * (num / den)

        return score

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform hybrid dense + sparse retrieval."""
        k = top_k or self.top_k
        dense_results = self.vector_store.similarity_search(query, top_k=k * 2)

        if not dense_results:
            return []

        # Rerank dense results using hybrid scoring (0.6 * dense + 0.4 * BM25)
        hybrid_results = []
        for item in dense_results:
            dense_score = item.get("score", 0.5)
            bm25_score = self._bm25_score(query, item["content"])
            
            # Normalize BM25 score
            norm_bm25 = min(1.0, bm25_score / 5.0)
            combined_score = (0.6 * dense_score) + (0.4 * norm_bm25)
            
            hybrid_results.append({
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "score": combined_score,
                "dense_score": dense_score,
                "bm25_score": bm25_score
            })

        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = hybrid_results[:k]
        logger.info(f"Retrieved top-{len(final_results)} hybrid results for query '{query[:30]}'")
        return final_results
