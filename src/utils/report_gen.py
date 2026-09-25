import os
from pathlib import Path
from typing import Optional
from config import OUTPUTS_DIR
from src.utils.logger import logger

def save_markdown_report(title: str, content: str, filename: Optional[str] = None) -> str:
    """Save report as a Markdown file."""
    if not filename:
        clean_title = "".join([c if c.isalnum() else "_" for c in title]).lower()[:30]
        filename = f"research_report_{clean_title}.md"
    
    filepath = OUTPUTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Report saved to Markdown: {filepath}")
    return str(filepath)

def export_to_pdf(markdown_content: str, title: str = "Research Report", filename: Optional[str] = None) -> str:
    """Export markdown report to PDF using FPDF2 with fallback."""
    if not filename:
        clean_title = "".join([c if c.isalnum() else "_" for c in title]).lower()[:30]
        filename = f"report_{clean_title}.pdf"
    
    filepath = OUTPUTS_DIR / filename
    
    try:
        from fpdf import FPDF
        
        class PDFReport(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.cell(0, 10, title[:60], border=False, new_x="LMARGIN", new_y="NEXT", align="C")
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        
        # Simple line-by-line conversion of markdown content
        lines = markdown_content.split("\n")
        for line in lines:
            line_str = line.encode("latin-1", "replace").decode("latin-1")
            if line.startswith("# "):
                pdf.set_font("Helvetica", "B", 16)
                pdf.multi_cell(0, 8, line_str[2:])
                pdf.ln(2)
            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.multi_cell(0, 7, line_str[3:])
                pdf.ln(2)
            elif line.startswith("### "):
                pdf.set_font("Helvetica", "B", 11)
                pdf.multi_cell(0, 6, line_str[4:])
                pdf.ln(1)
            elif line.startswith("- ") or line.startswith("* "):
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 5, f"  * {line_str[2:]}")
            else:
                pdf.set_font("Helvetica", size=10)
                if line_str.strip():
                    pdf.multi_cell(0, 5, line_str)
                else:
                    pdf.ln(3)

        pdf.output(str(filepath))
        logger.info(f"PDF export successful: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.warning(f"FPDF conversion failed ({e}). Falling back to saving Markdown file.")
        return save_markdown_report(title, markdown_content)
