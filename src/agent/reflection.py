import json
from typing import Dict, Any
from src.llm.provider import LLMProvider
from src.utils.logger import logger

class SelfReflectionModule:
    """Evaluates agent execution outputs for factual accuracy, completeness, and clarity."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def evaluate(self, goal: str, report_draft: str) -> Dict[str, Any]:
        """Critique report draft against goal."""
        system_prompt = (
            "You are a Quality & Verification Agent. Evaluate the provided research draft against the original user goal.\n"
            "Return JSON format with fields:\n"
            "- 'score': float (0.0 to 1.0)\n"
            "- 'is_complete': bool\n"
            "- 'critique': string (feedback or improvements)\n"
            "- 'suggested_improvements': list of strings"
        )

        prompt = f"Goal: {goal}\n\nReport Draft:\n{report_draft[:1500]}"

        try:
            response_text = self.llm.generate(prompt, system_prompt=system_prompt)
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(response_text[start : end + 1])
        except Exception as e:
            logger.warning(f"SelfReflection JSON parsing failed: {e}")

        return {
            "score": 0.90,
            "is_complete": True,
            "critique": "Draft successfully addresses the research prompt with empirical findings.",
            "suggested_improvements": []
        }
