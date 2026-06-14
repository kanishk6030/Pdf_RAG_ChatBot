import langchain
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, JINA_API_KEY
import streamlit as st

from langchain_openai import OpenAIEmbeddings

## This is not used because I am deploying the app on the Render and the sentence-transformers takes so much space that is not provide in the free tier 

# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL
#     )
import requests
from langchain_core.embeddings import Embeddings

class JinaEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = "jina-embeddings-v5-text-small"

    def embed_documents(self, texts):
        texts = [
            str(t).strip()
            for t in texts
            if t and str(t).strip()
        ]

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "task": "retrieval.passage",
                "normalized": True,
                "input": texts,
            },
        )

        response.raise_for_status()

        data = response.json()["data"]

        return [item["embedding"] for item in data]

    def embed_query(self, text):
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "task": "retrieval.query",
                "normalized": True,
                "input": [text],
            },
        )

        response.raise_for_status()

        return response.json()["data"][0]["embedding"]

def get_embeddings():
     return JinaEmbeddings(api_key=JINA_API_KEY)

@st.cache_resource
def load_embeddings():
    return get_embeddings()

