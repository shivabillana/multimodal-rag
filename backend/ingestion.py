import fitz  # PyMuPDF
import base64
import requests
from PIL import Image
import io
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from backend.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    VECTOR_SIZE,
    DPI,
    MAX_PAGES
)
import uuid

# Clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

def ensure_collection():
    """Create Qdrant collection if it doesn't exist."""
    from qdrant_client.models import PayloadSchemaType

    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists.")

    # Ensure filename index exists for filtering
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="filename",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print(f"[Qdrant] Filename index ensured.")


def pdf_to_images(pdf_path: str) -> list[dict]:
    """Convert PDF pages to base64 encoded images."""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(min(len(doc), MAX_PAGES)):
        page = doc[page_num]
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        pages.append({
            "page_number": page_num + 1,
            "image_base64": img_base64,
        })

    doc.close()
    print(f"[Ingestion] Converted {len(pages)} pages to images.")
    return pages


def embed_images(pages: list[dict]) -> list[dict]:
    """Embed each page image using OpenRouter Nemotron multimodal embeddings."""

    for i, page in enumerate(pages):
        payload = {
            "model": EMBEDDING_MODEL,
            "input": [
                {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{page['image_base64']}"
                            }
                        }
                    ]
                }
            ],
            "encoding_format": "float"
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=OPENROUTER_HEADERS,
            json=payload
        )

        result = response.json()

        if "data" in result:
            page["embedding"] = result["data"][0]["embedding"]
            print(f"[Ingestion] Embedded page {i+1}/{len(pages)}")
        else:
            print(f"[Ingestion] Error on page {i+1}: {result}")

    print(f"[Ingestion] Embedded {len(pages)} pages.")
    return pages

def store_in_qdrant(pages: list[dict], filename: str):
    """Store embedded pages in Qdrant one by one."""
    for page in pages:
        if "embedding" not in page:
            print(f"[Qdrant] Skipping page {page['page_number']} — no embedding.")
            continue

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=page["embedding"],
            payload={
                "filename": filename,
                "page_number": page["page_number"],
                "image_base64": page["image_base64"],
            }
        )

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        print(f"[Qdrant] Stored page {page['page_number']}/{len(pages)}")

    print(f"[Qdrant] Done storing {len(pages)} pages.")


def ingest_pdf(pdf_path: str, filename: str):
    """Full ingestion pipeline."""
    print(f"[Ingestion] Starting: {filename}")
    ensure_collection()
    pages = pdf_to_images(pdf_path)
    pages = embed_images(pages)
    store_in_qdrant(pages, filename)
    print(f"[Ingestion] Done: {filename}")
    return len(pages)