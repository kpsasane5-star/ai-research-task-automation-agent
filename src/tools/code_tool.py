import sys
import io
import traceback
from typing import Dict, Any
from config import CODE_TIMEOUT_SECONDS, OUTPUTS_DIR
from src.tools.base import BaseTool
from src.utils.logger import logger

class PythonCodeExecutorTool(BaseTool):
    """Tool for safely executing Python code to perform data analysis, math computations, and generate plots."""

    name = "code_executor"
    description = "Execute Python code snippets for mathematical calculations, data analysis with pandas/numpy, or matplotlib charts."

    def execute(self, query_or_input: str, **kwargs) -> Dict[str, Any]:
        logger.info("Executing PythonCodeExecutorTool...")
        
        # Clean code block indicators if passed by LLM
        code = query_or_input.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # Capture stdout and stderr
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        global_scope = {
            "OUTPUTS_DIR": str(OUTPUTS_DIR),
            "__name__": "__main__"
        }

        exec_status = "success"
        output_text = ""

        try:
            exec(code, global_scope)
            output_text = redirected_output.getvalue().strip()
            if not output_text:
                output_text = "Code executed successfully with no stdout output."
        except Exception as e:
            exec_status = "error"
            output_text = f"Execution Error: {str(e)}\n\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout

        return {
            "status": exec_status,
            "output": output_text,
            "executed_code": code
        }
