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

    # Split the documents into chunks
    chunks = splitter.split_documents(docs)
    
    # THE FIX: Filter out any chunks that have empty or None text.
    # This prevents the "Unprocessable Entity" error when sending data to the Jina/OpenAI embeddings API.
    # valid_chunks = [
    #     chunk for chunk in chunks 
    #     if chunk.page_content and chunk.page_content.strip()
    # ]

    return chunks