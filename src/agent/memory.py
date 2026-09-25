from typing import List, Dict, Any

class AgentMemory:
    """Working and short-term memory manager for the Agent."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.observations: List[Dict[str, Any]] = []
        self.findings: List[str] = []

    def add_step(self, thought: str, action: str, action_input: str, result: Any):
        """Record an execution step."""
        step = {
            "step": len(self.history) + 1,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "result": result
        }
        self.history.append(step)

    def add_finding(self, finding: str):
        """Record key factual finding."""
        if finding and finding not in self.findings:
            self.findings.append(finding)

    def get_summary_context(self) -> str:
        return self.get_history_summary()

    def get_history_summary(self) -> str:
        """Format history for LLM context inclusion."""
        if not self.history:
            return "No previous actions taken."

        summary_lines = []
        for h in self.history:
            res_str = str(h['result'])
            if len(res_str) > 300:
                res_str = res_str[:300] + "... [truncated]"
            summary_lines.append(
                f"Step {h['step']}: Action={h['action']}, Input='{h['action_input']}' -> Output: {res_str}"
            )
        return "\n".join(summary_lines)

    def clear(self):
        """Reset memory."""
        self.history.clear()
        self.observations.clear()
        self.findings.clear()
