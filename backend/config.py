import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Qdrant
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_rag")

# Ingestion
DPI = 150  # PDF to image resolution
MAX_PAGES = 50  # max pages per PDF

# Retrieval
TOP_K = 3  # number of pages to retrieve

# Vision LLM
OPENROUTER_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VECTOR_SIZE = 2048
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"