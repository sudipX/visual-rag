import os
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from ingestion.ingest import ingest_pdf
from query.retriever import retrieve_relevant_context
from query.llm_caller import answer_question

UPLOAD_DIR = Path("./uploads")
DB_PATH = "./chroma_db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf"}


class IngestResponse(BaseModel):
    source_filename: str
    pages_processed: int
    text_chunks_stored: int
    images_stored: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Your question about the uploaded documents"
    )
    top_k_text: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of text chunks to retrieve"
    )
    top_k_images: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Number of images to retrieve"
    )


class SourceReference(BaseModel):
    type: str
    source_number: int
    page_number: Optional[int]
    source_filename: Optional[str]
    relevance_score: float
    excerpt: Optional[str] = None
    image_path: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: initialising ChromaDB...")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    app.state.text_collection = chroma_client.get_or_create_collection(
        name="text_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    app.state.image_collection = chroma_client.get_or_create_collection(
        name="image_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    print(f"ChromaDB ready.")
    print(f"text_chunks: {app.state.text_collection.count()} items")
    print(f" image_chunks: {app.state.image_collection.count()} items")

    yield

    print("Shutting down...")


app = FastAPI(
    title="Visual RAG Pipeline API",
    description="Upload PDFs and ask questions grounded in your documents.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status":  "healthy",
        "message": "Visual RAG Pipeline API is running.",
        "docs":    "Visit /docs for interactive API documentation."
    }


@app.get("/stats")
async def get_stats():
    return {
        "text_chunks":  app.state.text_collection.count(),
        "image_chunks": app.state.image_collection.count()
    }


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document"
)
async def ingest_document(
    file: UploadFile = File(..., description="A PDF file to ingest")
):
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files accepted. Received: '{file_ext}'"
        )

    # Save uploaded file to disk
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as dest:
            shutil.copyfileobj(file.file, dest)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # Run ingestion pipeline
    try:
        summary = ingest_pdf(
            pdf_path=str(save_path),
            text_collection=app.state.text_collection,
            image_collection=app.state.image_collection
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

    return IngestResponse(
        source_filename=summary["source_filename"],
        pages_processed=summary["pages_processed"],
        text_chunks_stored=summary["text_chunks_stored"],
        images_stored=summary["images_stored"],
        message=f"Successfully ingested '{file.filename}'"
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about your uploaded documents"
)
async def query_documents(request: QueryRequest):
    # Guard: nothing to search if database is empty
    if app.state.text_collection.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents ingested yet. Upload a PDF using POST /ingest first."
        )

    # Retrieve relevant context from ChromaDB
    try:
        context = retrieve_relevant_context(
            question=request.question,
            text_collection=app.state.text_collection,
            image_collection=app.state.image_collection,
            top_k_text=request.top_k_text,
            top_k_images=request.top_k_images
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )

    try:
        result = answer_question(
            question=request.question,
            text_results=context["text_results"],
            image_results=context["image_results"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM call failed: {str(e)}"
        )

    source_refs = [SourceReference(**src) for src in result["sources"]]

    return QueryResponse(
        answer=result["answer"],
        sources=source_refs,
        question=request.question
    )