import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from backend.config import (
    OPENROUTER_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    TOP_K,
    EMBEDDING_MODEL
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

def embed_query(query: str) -> list[float]:
    """Embed a text query using OpenRouter."""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [query],
        "encoding_format": "float"
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers=OPENROUTER_HEADERS,
        json=payload
    )

    result = response.json()

    if "data" in result:
        return result["data"][0]["embedding"]
    else:
        raise ValueError(f"Embedding error: {result}")


def retrieve_pages(query: str, filenames: list[str] = None) -> list[dict]:
    """Retrieve top K most relevant pages for a query."""
    print(f"[Retrieval] Embedding query...")
    query_embedding = embed_query(query)

    print(f"[Retrieval] Searching Qdrant...")

    # Build filter if filenames provided
    search_filter = None
    if filenames:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchAny(any=filenames)
                )
            ]
        )

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=TOP_K,
        with_payload=True,
        query_filter=search_filter
    )

    pages = []
    for point in response.points:
        payload = point.payload
        score = point.score
        pages.append({
            "filename": payload["filename"],
            "page_number": payload["page_number"],
            "image_base64": payload["image_base64"],
            "score": score
        })
        print(f"[Retrieval] Found: {payload['filename']} page {payload['page_number']} (score: {score:.3f})")

    return pages