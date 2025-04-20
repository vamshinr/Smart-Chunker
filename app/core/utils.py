import fitz  # PyMuPDF
from io import BytesIO
import json
import os

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

def export_chunks_to_jsonl(chunks, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

