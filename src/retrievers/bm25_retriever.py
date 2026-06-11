from langchain_community.retrievers import BM25Retriever

def get_bm25_retriever(
    split_docs
):

    retriever = BM25Retriever.from_documents(
        split_docs
    )

    retriever.k = 4

    return retriever