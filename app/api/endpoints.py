from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.core.chunking import SmartChunker
from app.core.utils import extract_text_and_elements_from_pdf, export_chunks_to_jsonl
import uuid
import os

router = APIRouter()

class ChunkRequest(BaseModel):
    text: Optional[str] = None
    method: str = "semantic"
    chunk_size: int = 512
    overlap: int = 50
    export: bool = False

class Chunk(BaseModel):
    id: int
    text: str
    start_char: int
    end_char: int
    metadata: Optional[dict] = None

class ChunkResponse(BaseModel):
    num_chunks: int
    chunks: List[Chunk]
    export_file: Optional[str] = None

@router.post("/chunk", response_model=ChunkResponse)
def chunk_document(request: ChunkRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    chunker = SmartChunker(method=request.method, chunk_size=request.chunk_size, overlap=request.overlap)
    chunks = chunker.chunk_text(request.text, visual_elements=[])
    export_file = None
    if request.export:
        export_file = f"data/export/chunks_{uuid.uuid4().hex}.jsonl"
        export_chunks_to_jsonl(chunks, export_file)
    return ChunkResponse(
        num_chunks=len(chunks),
        chunks=[Chunk(id=i, text=c['text'], start_char=c['start'], end_char=c['end'], metadata=c.get('metadata')) for i, c in enumerate(chunks)],
        export_file=export_file
    )

@router.post("/chunk/file", response_model=ChunkResponse)
def chunk_file(
    file: UploadFile = File(...),
    method: str = Form("semantic"),
    chunk_size: int = Form(512),
    overlap: int = Form(50),
    export: bool = Form(False)
):
    contents = file.file.read()
    parsed = extract_text_and_elements_from_pdf(contents)
    text = parsed["text"]
    visual_elements = parsed["elements"]
    chunker = SmartChunker(method=method, chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_text(text, visual_elements=visual_elements)
    export_file = None
    if export:
        export_file = f"data/export/chunks_{uuid.uuid4().hex}.jsonl"
        export_chunks_to_jsonl(chunks, export_file)
    return ChunkResponse(
        num_chunks=len(chunks),
        chunks=[Chunk(id=i, text=c['text'], start_char=c['start'], end_char=c['end'], metadata=c.get('metadata')) for i, c in enumerate(chunks)],
        export_file=export_file
    )
