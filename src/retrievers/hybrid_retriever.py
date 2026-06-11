from langchain_classic.retrievers import EnsembleRetriever

from src.retrievers.bm25_retriever import (
    get_bm25_retriever
)

from src.retrievers.faiss_retriever import (
    get_faiss_retriever
)

def get_hybrid_retriever(
    split_docs,
    vectorstore
):

    bm25 = get_bm25_retriever(
        split_docs
    )

    faiss = get_faiss_retriever(
        vectorstore
    )

    return EnsembleRetriever(
        retrievers=[bm25,faiss],
        weights=[0.5,0.5]
    )