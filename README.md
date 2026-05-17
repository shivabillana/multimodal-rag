# 📄 Multimodal Document Intelligence

An AI-powered document Q&A system that goes beyond basic text extraction — it understands **charts, tables, diagrams, and visual content** inside PDFs using visual embeddings and a vision language model.

Built with **FastAPI**, **Qdrant**, **OpenRouter**, and **Streamlit**. Deployed on **Render** (API) and **Streamlit Cloud** (UI).

---

## 🧠 Why This Is Different From Basic Doc Q&A

Every standard RAG project does the same thing: extract text → chunk it → embed it → retrieve it. That approach **fails completely** on documents with:

- Revenue charts and financial graphs
- Tables with structured data
- Research paper diagrams and figures
- Slide decks with visual layouts

**This project treats each PDF page as an image.** Visual embeddings capture what text extraction misses — the entire page including all visual content — and a vision LLM reads the actual page images to answer questions.

---

## 🏗️ Architecture

```
INGESTION PIPELINE
PDF Uploaded
   ↓
PyMuPDF → converts each page to image (PNG)
   ↓
OpenRouter Embedding API (nvidia/llama-nemotron-embed-vl-1b-v2)
→ visual embedding per page (2048-dim vector)
   ↓
Qdrant Cloud → stores vector + page image + metadata

QUERY PIPELINE
User Question
   ↓
OpenRouter Embedding API → embed query as vector
   ↓
Qdrant → vector similarity search (filtered by session filenames)
→ top 3 most relevant page images retrieved
   ↓
OpenRouter Vision LLM (nvidia/nemotron-nano-12b-v2-vl)
→ reads retrieved page images + answers question
   ↓
Answer returned with page citations
```

---

## 🚀 Features

- **Visual PDF ingestion** — every page converted to image, embedded visually (not just text)
- **Multimodal retrieval** — search finds relevant pages based on visual + semantic similarity
- **Vision LLM answering** — model reads actual page images, understands charts and tables
- **Multi-document sessions** — upload multiple PDFs, queries filter to your session's documents only
- **FastAPI backend** — real REST API with `/upload` and `/query` endpoints
- **Streamlit frontend** — clean chat UI with session management
- **Fully deployed** — live API on Render, live UI on Streamlit Cloud

---

## 🛠️ Tech Stack

| Layer             | Technology                                  |
| ----------------- | ------------------------------------------- |
| PDF → Images      | PyMuPDF                                     |
| Visual Embeddings | OpenRouter (nvidia/llama-nemotron-embed-vl) |
| Vector Database   | Qdrant Cloud                                |
| Vision LLM        | OpenRouter (nvidia/nemotron-nano-12b-v2-vl) |
| Backend API       | FastAPI                                     |
| Frontend          | Streamlit                                   |
| API Deployment    | Render                                      |
| UI Deployment     | Streamlit Cloud                             |

---

## 📂 Project Structure

```
multimodal-rag/
│
├── backend/
│   ├── config.py        # env vars and settings
│   ├── ingestion.py     # PDF → images → embeddings → Qdrant
│   ├── retrieval.py     # query embedding + Qdrant search with filename filter
│   ├── generation.py    # vision LLM answer generation
│   └── main.py          # FastAPI app with /upload and /query endpoints
│
├── frontend/
│   └── app.py           # Streamlit chat UI
│
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/shivabillana/multimodal-rag.git
cd multimodal-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=multimodal_rag
```

### 4. Run the API

```bash
uvicorn backend.main:app --reload
```

API runs at: `http://localhost:8000`
Swagger docs: `http://localhost:8000/docs`

### 5. Run the frontend

```bash
streamlit run frontend/app.py
```

UI runs at: `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint  | Description                                           |
| ------ | --------- | ----------------------------------------------------- |
| GET    | `/`       | Health check                                          |
| POST   | `/upload` | Upload a PDF — ingests all pages as visual embeddings |
| POST   | `/query`  | Ask a question, filtered to specific uploaded files   |

### Query request format

```json
{
  "question": "What was the revenue growth in Q3?",
  "filenames": ["annual_report.pdf"]
}
```

---

## 🔑 Key Technical Decisions

**Why visual embeddings over text extraction?**
Text extraction fails silently on charts, diagrams, and visual data. Visual embeddings treat the entire page as a unified representation — text, layout, and visuals all captured together.

**Why store images in Qdrant payload?**
Avoids a separate file storage system for the portfolio demo. In production, images would live in S3/Supabase with only URLs stored in Qdrant.

**Why filter by filename instead of per-session collections?**
Simpler architecture with the same result — each session tracks its uploaded filenames and queries are filtered accordingly. No collection creation/deletion overhead.

**Why OpenRouter for both embeddings and generation?**
Single API key, multiple free models, no GPU required. The `nvidia/llama-nemotron-embed-vl` model is purpose-built for visual document retrieval.

---

## 🔮 Future Improvements

- Hybrid retrieval — combine visual similarity with keyword search
- Answer citations — highlight exactly which part of the page was used
- Streaming responses — real-time answer generation
- Cloud file storage — S3/Supabase for production-scale image storage
- Evaluation pipeline — automated answer quality scoring per query

---

## 👤 Author

Shiva Kumar Billana
