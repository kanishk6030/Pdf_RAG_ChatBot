from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL
import streamlit as st

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

@st.cache_resource
def load_embeddings():
    return get_embeddings()

