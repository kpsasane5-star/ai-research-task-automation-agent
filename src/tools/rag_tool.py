from typing import Dict, Any, Optional
from src.tools.base import BaseTool
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStoreManager
from src.utils.logger import logger

class DocumentRAGTool(BaseTool):
    """Tool for retrieving relevant knowledge from uploaded documents via Hybrid RAG."""

    name = "rag_query"
    description = "Search and extract knowledge from uploaded PDF/text documents in the vector database."

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        if not retriever:
            vector_store = VectorStoreManager()
            self.retriever = HybridRetriever(vector_store)
        else:
            self.retriever = retriever

    def execute(self, query_or_input: str, top_k: int = 4, **kwargs) -> Dict[str, Any]:
        logger.info(f"Executing DocumentRAGTool for query: '{query_or_input}'")
        results = self.retriever.retrieve(query_or_input, top_k=top_k)

        if not results:
            return {
                "status": "empty",
                "output": "No relevant document passages found in the RAG knowledge base. Try uploading documents first or refining your query.",
                "passages": []
            }

        passages_text = []
        for idx, res in enumerate(results):
            source = res["metadata"].get("source", "document")
            score = round(res.get("score", 0.0), 3)
            passages_text.append(f"[Source {idx+1}: {source} (relevance score: {score})]\n{res['content']}")

        formatted_output = "\n\n".join(passages_text)

        return {
            "status": "success",
            "count": len(results),
            "output": formatted_output,
            "passages": results
        }
