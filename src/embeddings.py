from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )