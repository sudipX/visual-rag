# Visual RAG Pipeline

A multimodal Retrieval-Augmented Generation system built from scratch : no LangChain, no abstractions. Upload PDFs, ask questions, get grounded answers with cited sources.

---

## What It Does

Most LLMs cannot access your private documents. Ask Claude or any model about a PDF you just uploaded, and it either admits it doesn't know or confidently makes something up.

This system solves that. When you upload a PDF, it extracts the text and images, converts them into meaning-encoded vectors (embeddings), and stores them in a searchable vector database. When you ask a question, it finds the most relevant content, hands it to an LLM as context, and returns an answer grounded in your actual document, with page-level source citations.

Tested on a real Autonomous Vehicle System report: the system correctly retrieved and cited specific technical details across 39 pages, with 45 text chunks and 5 embedded images stored and searchable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION  (once per PDF)                  │
│                                                                  │
│  PDF Upload → Extract Text & Images → Embed → Store in ChromaDB │
│               (pdfplumber)    (CLIP/ST)        (cosine index)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY  (every request)                     │
│                                                                  │
│  Question → Embed → Search ChromaDB → Assemble Prompt → LLM     │
│             (ST)    (top-K cosine)    (grounded)       (Ollama)  │
│                                                    ↓             │
│                                          Answer + Citations      │
└─────────────────────────────────────────────────────────────────┘
```

**Two separate embedding models, two separate collections:**
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim) : text chunks
- `openai/clip-vit-base-patch32` (512-dim) : images

CLIP's cross-modal property means a text query like *"show me the architecture diagram"* can retrieve relevant images purely through semantic similarity : no OCR, no captions required.

---

## Tech Stack

| Component | Tool |
|---|---|
| Web Framework | FastAPI |
| Vector Database | ChromaDB (persistent, cosine similarity) |
| Text Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Image Embeddings | CLIP via HuggingFace Transformers |
| PDF Processing | pdfplumber |
| LLM | Ollama (llama3.2:3b, runs locally) |
| Validation | Pydantic v2 |

**No LangChain. No paid APIs. Runs entirely locally.**

---

## Project Structure

```
visual-rag/
├── main.py                    # FastAPI app: routes, lifespan, middleware
├── models.py                  # Pydantic request/response schemas
├── requirements.txt
├── .env                       # API keys (gitignored)
│
├── embeddings/
│   ├── clip_embedder.py       # CLIP: image + short-text embeddings (512-dim)
│   └── text_embedder.py       # sentence-transformers: text embeddings (384-dim)
│
├── ingestion/
│   ├── pdf_extractor.py       # PDF text and image extraction
│   ├── chunker.py             # Word-level chunking with overlap (400 words, 80 overlap)
│   └── ingest.py              # Full ingestion pipeline
│
├── query/
│   ├── retriever.py           # ChromaDB similarity search
│   ├── prompt_builder.py      # Grounded prompt assembly with source numbering
│   └── llm_caller.py          # Ollama API call
│
├── uploads/                   # Uploaded PDFs (auto-created)
└── chroma_db/                 # Vector database files (auto-created, persistent)
```

---

## Setup and Installation

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed and running

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/visual-rag.git
cd visual-rag
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Pull the Ollama model**
```bash
ollama pull llama3.2:3b
```

**5. Create your `.env` file**
```bash
# .env
# No API keys required — this project runs fully locally with Ollama
```

**6. Start the server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Using the API

### Option A — Interactive UI (recommended for first use)

Open your browser at `http://localhost:8000/docs` : FastAPI's auto-generated Swagger UI lets you upload files and send queries directly from the browser.

### Option B — curl

**Ingest a PDF:**
```bash
curl -X POST http://localhost:8000/ingest \
     -F "file=@/path/to/your/document.pdf"
```

**Example response:**
```json
{
  "source_filename": "Autonomous_vehicle_system.pdf",
  "pages_processed": 39,
  "text_chunks_stored": 45,
  "images_stored": 5,
  "message": "Successfully ingested 'Autonomous_vehicle_system.pdf'"
}
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What sensors does the autonomous vehicle use for obstacle detection?"}'
```

**Example response:**
```json
{
  "answer": "According to [TEXT SOURCE 1], the autonomous vehicle uses ultrasonic sensors for obstacle detection, integrated with a Raspberry Pi for real-time processing...",
  "sources": [
    {
      "type": "text",
      "source_number": 1,
      "page_number": 4,
      "source_filename": "Autonomous_vehicle_system.pdf",
      "relevance_score": 0.9134,
      "excerpt": "The obstacle avoidance system uses ultrasonic sensors..."
    }
  ],
  "question": "What sensors does the autonomous vehicle use for obstacle detection?"
}
```

**Check database stats:**
```bash
curl http://localhost:8000/stats
```

### Ingest multiple documents

Each ingestion call adds to the existing database without overwriting. Query across all documents simultaneously:

```bash
curl -X POST http://localhost:8000/ingest -F "file=@document1.pdf"
curl -X POST http://localhost:8000/ingest -F "file=@document2.pdf"
curl -X POST http://localhost:8000/ingest -F "file=@document3.pdf"
# All three are now searchable in a single query
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/stats` | Item counts in ChromaDB collections |
| `POST` | `/ingest` | Upload and ingest a PDF |
| `POST` | `/query` | Ask a question about ingested documents |

### Query parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Your question (3–1000 characters) |
| `top_k_text` | int | 5 | Text chunks to retrieve (1–10) |
| `top_k_images` | int | 2 | Images to retrieve (0–5) |

---

## Design Decisions

**Why no LangChain?** LangChain abstracts away exactly the layers that matter for understanding: how embeddings are compared, how the prompt is constructed, why retrieval works or fails. Every component here is written from scratch so the system is fully debuggable and every decision is explicit.

**Why two embedding models?** CLIP embeds images and text into the same vector space (cross-modal retrieval). sentence-transformers is optimised for longer text passages (up to 256 word-pieces vs CLIP's 77-token limit). Each model does what it's best at.

**Why chunking with overlap?** A 400-word chunk embedded as one vector gives the retriever a focused, specific signal. Full pages produce blurry, topic-averaged embeddings that score poorly against specific questions. 80-word overlap prevents important sentences from being split across two chunks where they might be missed entirely.

**Why explicit prompt grounding?** Without the instruction "use ONLY the provided context," the LLM mixes its own training knowledge with retrieved content — making it impossible to verify whether an answer is actually backed by your document.

**Data persistence:** ChromaDB uses a `PersistentClient` : all embeddings are written to disk in `chroma_db/`. Restarting the server does not lose any ingested documents. Ingesting the same PDF twice is safe — `.upsert()` replaces existing chunks by ID rather than creating duplicates.

---

## Known Limitations

- **Scanned PDFs** (images of text rather than digital text) return no extractable text. OCR support would require an additional library such as Tesseract.
- **Image comprehension:** Images are retrieved by semantic similarity (CLIP finds which image is relevant to your query) but the LLM does not receive the image's pixel content, it only knows an image exists at a given page. True image-grounded answers would require a multimodal LLM call.
- **In-process only:** The server must remain running to serve queries. The vector data persists on disk; the server process itself does not.

---

## Requirements

```
fastapi
uvicorn[standard]
chromadb
sentence-transformers
transformers
torch
torchvision
Pillow
pdfplumber
python-multipart
python-dotenv
numpy
ollama
```