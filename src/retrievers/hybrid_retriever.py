from langchain_classic.retrievers import EnsembleRetriever

from src.retrievers.bm25_retriever import (
    get_bm25_retriever
)

from src.retrievers.chroma_retriever import (
    get_chroma_retriever
)

from evaluation.retrieval_eval import (
    evaluate_retriever
)

def get_hybrid_retriever(
    split_docs,
    vectorstore
):

    bm25 = get_bm25_retriever(
        split_docs
    )

    chroma = get_chroma_retriever(
        vectorstore
    )

    evaluate_retriever(chroma);

    return EnsembleRetriever(
        retrievers=[bm25,chroma],
        weights=[0.5,0.5]
    )