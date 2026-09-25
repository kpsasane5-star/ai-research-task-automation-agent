import json
from typing import Dict, Any, List, Optional, Callable
from config import MAX_ITERATIONS
from src.llm.provider import LLMProvider, get_llm_provider
from src.agent.memory import AgentMemory
from src.agent.planner import GoalPlanner
from src.agent.reflection import SelfReflectionModule
from src.tools.base import BaseTool
from src.tools.search_tool import WebSearchTool
from src.tools.rag_tool import DocumentRAGTool
from src.tools.code_tool import PythonCodeExecutorTool
from src.tools.file_tool import ReportExporterTool
from src.utils.logger import logger

class AgentExecutor:
    """Core Agentic Execution Engine implementing ReAct loop with memory, tools, and reflection."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.llm = llm_provider or get_llm_provider()
        self.memory = AgentMemory()
        self.planner = GoalPlanner(self.llm)
        self.reflection = SelfReflectionModule(self.llm)
        self.max_iterations = max_iterations

        # Register tools
        default_tools = [
            WebSearchTool(),
            DocumentRAGTool(),
            PythonCodeExecutorTool(),
            ReportExporterTool(),
        ]
        tool_list = tools or default_tools
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tool_list}

    def run(self, goal: str, callback: Optional[Callable[[str, Any], None]] = None) -> Dict[str, Any]:
        """Execute full autonomous agent pipeline for a goal."""
        self.memory.clear()
        logger.info(f"Starting Agent execution for goal: '{goal}'")
        if callback:
            callback("status", f"Decomposing goal into sub-tasks...")

        # Step 1: Goal Planning
        plan = self.planner.create_plan(goal)
        if callback:
            callback("plan", plan)

        collected_insights = []

        # Step 2: Iterate over sub-tasks
        for task in plan:
            task_title = task.get("title", "")
            tool_name = task.get("tool", "web_search")
            query = task.get("query", goal)

            logger.info(f"Executing Sub-task: {task_title} using [{tool_name}]")
            if callback:
                callback("status", f"Executing task: {task_title}")

            # ReAct Execution step
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    res = tool.execute(query)
                    out_text = res.get("output", str(res))
                    self.memory.add_step(
                        thought=f"Executing planned sub-task: {task_title}",
                        action=tool_name,
                        action_input=query,
                        result=out_text,
                    )
                    collected_insights.append(f"### Sub-Task: {task_title}\n**Tool Used**: `{tool_name}`\n**Query**: {query}\n\n**Output**:\n{out_text}")
                    if callback:
                        callback("step", {
                            "task": task_title,
                            "tool": tool_name,
                            "output": out_text
                        })
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    self.memory.add_step(
                        thought=f"Execution error on {task_title}",
                        action=tool_name,
                        action_input=query,
                        result=f"Error: {e}"
                    )

        # Step 3: Report Synthesis
        if callback:
            callback("status", "Synthesizing comprehensive research report...")

        report_prompt = (
            f"User Goal: {goal}\n\n"
            f"Execution Steps & Observations:\n" + "\n\n".join(collected_insights) + "\n\n"
            "Synthesize a well-formatted, professional research report in Markdown. "
            "Include Title, Executive Summary, Key Findings, Technical Details, and Conclusion."
        )
        
        report_draft = self.llm.generate(
            prompt=report_prompt,
            system_prompt="You are a Principal AI Researcher synthesizing final technical reports."
        )

        # Step 4: Self Reflection & Critique
        if callback:
            callback("status", "Evaluating report draft with Self-Reflection Module...")
        
        eval_result = self.reflection.evaluate(goal, report_draft)

        # Step 5: Export Artifacts
        exporter = self.tools.get("report_exporter")
        export_res = {}
        if exporter:
            export_res = exporter.execute(report_draft, title=goal[:40])

        final_result = {
            "goal": goal,
            "plan": plan,
            "history": self.memory.history,
            "report_markdown": report_draft,
            "evaluation": eval_result,
            "exports": export_res
        }

        if callback:
            callback("complete", final_result)

        logger.info("Agent execution completed successfully.")
        return final_result
