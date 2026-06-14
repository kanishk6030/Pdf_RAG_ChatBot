#Dense retreiveal using Chroma with MMR (Maximal Marginal Relevance)
def get_chroma_retriever(
    vectorstore
):

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 15
        }
    )