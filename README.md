# Smart Chunker – AI-Optimized Document Chunking API

Smart Chunker is a Python FastAPI microservice for intelligently chunking long documents (PDFs, raw text) for LLM pipelines, vector DB ingestion, or retrieval-augmented generation (RAG) applications.

## Why Use This?
Naive chunking (e.g., every 512 tokens) often cuts off meaning or context. Smart Chunker supports:
- Fixed-window token chunking
- Header/section-aware chunking
- Semantic chunking using sentence embeddings

## Features
- Accepts text or PDF input
- Semantic chunking via `sentence-transformers`
- Optional overlap control
- Export to `.jsonl` format for vector DB ingestion (e.g., FAISS, Chroma)
- Clean FastAPI backend

## Quickstart
```bash
git clone https://github.com/yourname/smart-chunker
cd smart-chunker
pip install -r requirements.txt
uvicorn main:app --reload
