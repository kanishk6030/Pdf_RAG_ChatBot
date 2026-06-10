from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def load_and_split(pdf_path):

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(docs)