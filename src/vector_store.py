from langchain_chroma import Chroma

def create_vectorstore(documents, embeddings):

    # Dense Search (FAISS + MMR)
    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        # persist_directory="./chroma_db"
    )

    return vectorstore