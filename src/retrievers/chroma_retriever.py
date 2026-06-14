#Dense retreiveal using Chroma with MMR (Maximal Marginal Relevance)
def get_chroma_retriever(
    vectorstore
):

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3, # number of documents to return 6 for better results
            "fetch_k": 10 # number of documents to fetch for MMR re-ranking increse the value to get better results, but it will increase the latency
        }
    )