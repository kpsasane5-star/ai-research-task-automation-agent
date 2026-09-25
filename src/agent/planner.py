import json
from typing import List, Dict, Any
from src.llm.provider import LLMProvider
from src.utils.logger import logger

class GoalPlanner:
    """Decomposes complex user goals into an ordered task breakdown."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def create_plan(self, user_goal: str) -> List[Dict[str, Any]]:
        """Decompose user goal into sub-tasks with assigned tools."""
        system_prompt = (
            "You are an expert AI Research & Task Planning Agent. Your job is to break down a high-level user goal "
            "into 3 to 5 clear, sequential sub-tasks. Available tools for tasks: 'web_search', 'rag_query', "
            "'code_executor', 'report_exporter'.\n\n"
            "Return ONLY a JSON list of objects with fields: 'id' (int), 'title' (str), 'tool' (str), 'query' (str)."
        )
        
        user_prompt = f"Goal: {user_goal}\n\nDecompose this goal into sub-tasks."

        try:
            response_text = self.llm.generate(user_prompt, system_prompt=system_prompt)
            # Find JSON array
            start = response_text.find("[")
            end = response_text.rfind("]")
            if start != -1 and end != -1:
                json_str = response_text[start : end + 1]
                tasks = json.loads(json_str)
                logger.info(f"GoalPlanner generated {len(tasks)} sub-tasks.")
                return tasks
        except Exception as e:
            logger.warning(f"Planner JSON parsing failed ({e}). Using default plan.")

        # Default fallback plan
        return [
            {"id": 1, "title": "Perform initial web research", "tool": "web_search", "query": user_goal},
            {"id": 2, "title": "Extract domain knowledge from documents", "tool": "rag_query", "query": user_goal},
            {"id": 3, "title": "Run computational verification or code analysis", "tool": "code_executor", "query": "print('Analysis active')"},
            {"id": 4, "title": "Synthesize final research report", "tool": "report_exporter", "query": "Final Report"}
        ]
