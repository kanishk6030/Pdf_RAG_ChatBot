import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"