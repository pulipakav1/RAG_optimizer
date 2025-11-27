import io
from typing import List
from pypdf import PdfReader


def pdf_bytes_to_text(file_bytes: bytes) -> str:
    """
    Convert PDF bytes to raw text using pypdf.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def merge_texts(texts: List[str]) -> str:
    """
    Merge multiple document texts into one big string.
    """
    return "\n\n".join(texts)
