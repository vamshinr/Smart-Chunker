import fitz  # PyMuPDF
from io import BytesIO
import json
import os

def extract_text_and_elements_from_pdf(pdf_bytes: bytes):
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    full_text = ""
    visual_elements = []
    char_offset = 0

    for page_num, page in enumerate(doc):
        text_page = page.get_text("dict")
        page_text = ""

        for block in text_page["blocks"]:
            if "lines" not in block:
                continue
            block_text = " ".join(span["text"] for line in block["lines"] for span in line["spans"])
            if block_text:
                page_text += block_text + "\n"

        for img in page.get_images(full=True):
            visual_elements.append({"type": "image", "page": page_num, "start": char_offset})

        # Heuristic: detect code by monospaced font & high newline ratio
        for block in text_page["blocks"]:
            if "lines" in block:
                line_texts = [span["text"] for line in block["lines"] for span in line["spans"] if span.get("font", "").lower().startswith("cour")]
                if line_texts:
                    visual_elements.append({"type": "code", "page": page_num, "start": char_offset})

        full_text += page_text
        char_offset += len(page_text)

    return {"text": full_text, "elements": visual_elements}

def export_chunks_to_jsonl(chunks, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
