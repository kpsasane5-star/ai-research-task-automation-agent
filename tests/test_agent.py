import unittest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.provider import LLMProvider
from src.rag.ingest import DocumentIngestor
from src.rag.vector_store import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.tools.search_tool import WebSearchTool
from src.tools.code_tool import PythonCodeExecutorTool
from src.agent.executor import AgentExecutor

class TestAIResearchAgent(unittest.TestCase):

    def test_llm_provider_fallback(self):
        llm = LLMProvider(provider="gemini")
        res = llm.generate("Test query prompt")
        self.assertIsNotNone(res)
        self.assertTrue(len(res) > 0)

    def test_rag_ingest_and_retrieval(self):
        ingestor = DocumentIngestor()
        chunks = ingestor.process_document("Artificial Intelligence agents perform autonomous reasoning and execution loops.", source_name="test_doc.txt")
        self.assertTrue(len(chunks) > 0)

        vector_store = VectorStoreManager(collection_name="test_collection")
        vector_store.add_chunks(chunks)

        retriever = HybridRetriever(vector_store)
        results = retriever.retrieve("autonomous reasoning", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])

    def test_web_search_tool(self):
        tool = WebSearchTool()
        res = tool.execute("AI Agent technology 2026")
        self.assertEqual(res["status"], "success")
        self.assertIn("output", res)

    def test_code_executor_tool(self):
        tool = PythonCodeExecutorTool()
        code = "a = 10\nb = 20\nprint(f'Sum is {a + b}')"
        res = tool.execute(code)
        self.assertEqual(res["status"], "success")
        self.assertIn("Sum is 30", res["output"])

    def test_agent_executor_end_to_end(self):
        executor = AgentExecutor()
        result = executor.run("Research state of AI agents")
        self.assertIn("report_markdown", result)
        self.assertIsNotNone(result["report_markdown"])
        self.assertIn("evaluation", result)

if __name__ == "__main__":
    unittest.main()
