
# Smart Chunker — AI-Optimized Document Chunking API

Smart Chunker is a FastAPI-powered microservice that preprocesses large documents (PDF/text) into semantically meaningful chunks tailored for LLM workflows like RAG (retrieval-augmented generation), vector search, summarization, and more.
It supports novel and classical chunking strategies — including an adaptive semantic algorithm (SWRAX) — and detects visual structures such as images and code blocks to enrich metadata for downstream models.
---

## Features

Upload PDF or plain text
Export chunks in `.jsonl` format for vector DB ingestion  
Powered by sentence-transformers (`MiniLM-L6-v2`)  
Clean, modular Python backend (FastAPI)
- Choose from multiple chunking strategies:
  - `fixed`, `header`, `semantic`, `swarax`
  - Naive chunking (e.g., every 512 tokens) often cuts off meaning or context. Smart Chunker supports:
    - Fixed-window token chunking
    - Header/section-aware chunking
    - Semantic chunking using sentence embeddings
    - Annotate chunks with:
  - image presence
  - table detection
  - code detection

---

## Chunking Strategies

| Method     | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `fixed`    | Naive fixed-length character chunks with configurable overlap               |
| `header`   | Section-based splits using heading cues (`Section`, `Chapter`, `##`)        |
| `semantic` | Sentence-aware chunking based on token length, preserving sentence bounds   |
| `swarax`   | 🔬 Novel adaptive strategy that stops on semantic drift using cosine similarity of sentence embeddings |

---

## Stack

- **Backend**: FastAPI
- **Embeddings**: `sentence-transformers` (MiniLM)
- **PDF Parsing**: `PyMuPDF`
- **Tokenizer**: `nltk`
- **Data Export**: `.jsonl` for Chroma, FAISS, etc.

---

## Installation

```bash
git clone https://github.com/yourusername/smart-chunker
cd smart-chunker
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt

