from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as chunk_router

import uvicorn

app = FastAPI(title="Smart Chunker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chunk_router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
