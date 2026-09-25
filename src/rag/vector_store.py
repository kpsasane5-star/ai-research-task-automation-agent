import os
import uuid
from typing import List, Dict, Any, Optional
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from src.rag.ingest import DocumentChunk
from src.utils.logger import logger

class VectorStoreManager:
    """ChromaDB & SentenceTransformer vector store manager with in-memory fallback."""

    def __init__(self, collection_name: str = "ai_research_agent"):
        self.collection_name = collection_name
        self.chroma_client = None
        self.collection = None
        self.embedder = None
        self.in_memory_docs: List[Dict[str, Any]] = []
        
        self._init_embedding_model()
        self._init_chroma()

    def _init_embedding_model(self):
        """Initialize sentence-transformers model if installed."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Loaded SentenceTransformer embedding model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using lightweight tf-idf fallback.")

    def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"ChromaDB initialization fallback to memory: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        if self.embedder:
            return self.embedder.encode(text).tolist()
        # Fallback simple deterministic feature vector
        words = text.lower().split()
        return [float(hash(w) % 100) / 100.0 for w in words[:64]] + [0.0] * max(0, 64 - len(words))

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to vector database."""
        if not chunks:
            return

        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"{c.source}_{c.chunk_id}_{uuid.uuid4().hex[:6]}" for c in chunks]

        if self.collection:
            try:
                embeddings = [self.get_embedding(doc) for doc in documents]
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB.")
                return
            except Exception as e:
                logger.error(f"Failed adding chunks to ChromaDB: {e}")

        # In-memory storage fallback
        for idx, chunk in enumerate(chunks):
            self.in_memory_docs.append({
                "id": ids[idx],
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": self.get_embedding(chunk.content)
            })
        logger.info(f"Indexed {len(chunks)} chunks into in-memory vector store.")

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search vector database for query matches."""
        query_embedding = self.get_embedding(query)
        results = []

        if self.collection and self.collection.count() > 0:
            try:
                res = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.collection.count())
                )
                if res and res.get("documents"):
                    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                        results.append({"content": doc, "metadata": meta, "score": 0.85})
                    return results
            except Exception as e:
                logger.error(f"ChromaDB search query failed: {e}")

        # In-memory cosine similarity fallback
        if self.in_memory_docs:
            def cosine_sim(v1, v2):
                dot = sum(a * b for a, b in zip(v1, v2))
                m1 = sum(a * a for a in v1) ** 0.5
                m2 = sum(b * b for b in v2) ** 0.5
                return dot / (m1 * m2 + 1e-8)

            scored_docs = [
                {"content": item["content"], "metadata": item["metadata"], "score": cosine_sim(query_embedding, item["embedding"])}
                for item in self.in_memory_docs
            ]
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:top_k]

        return results
