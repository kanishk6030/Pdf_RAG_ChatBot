from langchain_community.vectorstores import FAISS

def create_vectorstore(documents, embeddings):

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    return vectorstore