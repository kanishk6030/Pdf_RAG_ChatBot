from langchain_community.vectorstores import FAISS

def create_vectorstore(documents, embeddings):

    # Dense Search (FAISS + MMR)
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    return vectorstore