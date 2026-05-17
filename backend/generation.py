import requests
import base64
from backend.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL
)

OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

def generate_answer(query: str, pages: list[dict]) -> str:
    """Pass retrieved page images to vision LLM and get an answer."""

    if not pages:
        return "No relevant pages found to answer your question."

    # Build content with all retrieved page images
    content = []

    # Add the question first
    content.append({
        "type": "text",
        "text": f"Answer this question based on the document pages provided: {query}"
    })

    # Add each retrieved page image
    for page in pages:
        content.append({
            "type": "text",
            "text": f"Page {page['page_number']} from {page['filename']}:"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{page['image_base64']}"
            }
        })

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=OPENROUTER_HEADERS,
        json=payload
    )

    result = response.json()

    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {result}"