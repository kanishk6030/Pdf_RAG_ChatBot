import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

# OPEN_API_KEY = os.getenv("OPEN_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "jina-embeddings-v3"
# EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "llama-3.3-70b-versatile"