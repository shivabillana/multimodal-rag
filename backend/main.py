import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from backend.ingestion import ingest_pdf, ensure_collection
from backend.retrieval import retrieve_pages
from backend.generation import generate_answer

app = FastAPI(title="Multimodal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    ensure_collection()

class QueryRequest(BaseModel):
    question: str
    filenames: List[str] = []

class QueryResponse(BaseModel):
    question: str
    answer: str
    pages_used: int

@app.get("/")
def root():
    return {"status": "Multimodal RAG API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pages = ingest_pdf(tmp_path, file.filename)
    finally:
        os.remove(tmp_path)

    return {
        "filename": file.filename,
        "pages_ingested": pages,
        "status": "success"
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    filenames = request.filenames if request.filenames else None
    pages = retrieve_pages(request.question, filenames)
    answer = generate_answer(request.question, pages)

    return QueryResponse(
        question=request.question,
        answer=answer,
        pages_used=len(pages)
    )