from typing import Dict, Any, Optional
from src.tools.base import BaseTool
from src.utils.report_gen import save_markdown_report, export_to_pdf
from src.utils.logger import logger

class ReportExporterTool(BaseTool):
    """Tool for exporting research reports to Markdown and PDF files."""

    name = "report_exporter"
    description = "Export generated research findings into Markdown (.md) and PDF (.pdf) files."

    def execute(self, query_or_input: str, title: str = "AI Research Synthesis", export_pdf: bool = True, **kwargs) -> Dict[str, Any]:
        logger.info(f"Executing ReportExporterTool for report: '{title}'")
        
        md_path = save_markdown_report(title, query_or_input)
        pdf_path = None
        
        if export_pdf:
            pdf_path = export_to_pdf(query_or_input, title=title)

        output_msg = f"Report successfully saved.\n- Markdown Path: {md_path}"
        if pdf_path:
            output_msg += f"\n- PDF Path: {pdf_path}"

        return {
            "status": "success",
            "markdown_path": md_path,
            "pdf_path": pdf_path,
            "output": output_msg
        }
