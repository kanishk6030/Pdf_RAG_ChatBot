#Dense retreiveal using FAISS with MMR (Maximal Marginal Relevance)
def get_faiss_retriever(
    vectorstore
):

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )