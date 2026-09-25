import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from config import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import logger

class DocumentChunk:
    def __init__(self, content: str, source: str, chunk_id: int, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.source = source
        self.chunk_id = chunk_id
        self.metadata = metadata or {}

class DocumentIngestor:
    """Document loader and recursive chunker for PDFs, Markdown, TXT, and CSV files."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: Union[str, Path]) -> str:
        """Read content from text, markdown, pdf, or csv file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = []
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text.append(f"--- Page {idx+1} ---\n{page_text}")
                return "\n".join(text)
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                return ""
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def chunk_text(self, text: str, source_name: str) -> List[DocumentChunk]:
        """Split text recursively into overlapping chunks."""
        if not text.strip():
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        text_length = len(text)
        chunk_idx = 0

        while start < text_length:
            end = start + self.chunk_size
            
            # Try to break at natural boundary (newline or period)
            if end < text_length:
                break_point = text.rfind("\n", start, end)
                if break_point == -1 or break_point <= start:
                    break_point = text.rfind(". ", start, end)
                if break_point != -1 and break_point > start:
                    end = break_point + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        source=source_name,
                        chunk_id=chunk_idx,
                        metadata={"source": source_name, "chunk_id": chunk_idx, "length": len(chunk_content)}
                    )
                )
                chunk_idx += 1

            start = end - self.chunk_overlap
            if start < 0 or start >= text_length:
                break

        logger.info(f"Generated {len(chunks)} chunks for source '{source_name}'")
        return chunks

    def process_document(self, file_path_or_content: Union[str, Path], source_name: Optional[str] = None) -> List[DocumentChunk]:
        """Load file or process raw text and return list of DocumentChunks."""
        if isinstance(file_path_or_content, (str, Path)) and os.path.exists(str(file_path_or_content)):
            path = Path(file_path_or_content)
            source = source_name or path.name
            text = self.load_file(path)
        else:
            source = source_name or "raw_text_input"
            text = str(file_path_or_content)

        return self.chunk_text(text, source)
