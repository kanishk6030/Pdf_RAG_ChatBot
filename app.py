import streamlit as st
import tempfile

from langchain_groq import ChatGroq

from config.settings import *

from src.retrievers.hybrid_retriever import get_hybrid_retriever
from src.embeddings import get_embeddings
from src.pdf_loader import load_and_split
from src.vector_store import create_vectorstore
from src.rag_chain import build_rag_chain

st.title("Conversational RAG Chatbot")

api_key = st.text_input(
    "Enter Groq API Key",
    type="password"
)

if api_key:

    llm = ChatGroq(
        api_key=api_key,
        model=LLM_MODEL
    )

    session_id = st.text_input(
        "Session ID",
        value="default"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        documents = []

        ## Upload the pdf and split into chunks and store in dcouments list
        for file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(file.getvalue())

                documents.extend(
                    load_and_split(tmp.name)
                )

        ## Create embeddings for the documents and store in vectorstore
        embeddings = get_embeddings()

        vectorstore = create_vectorstore(
            documents,
            embeddings
        )

        ## Make the vectorstore as retriever and build the RAG chain
        # Now we gonna use the Hybrid Search - Dense and Sparse search (MMR) for better results

## Hybrid Search - Ensemble of BM25 and FAISS with equal weights    
        retriever = get_hybrid_retriever(
            documents,
            vectorstore
        )

        chain = build_rag_chain(
            llm,
            retriever
        )

        ## Take user query as input and pass it to the chain and display the response
        query = st.text_input(
            "Ask Question"
        )

        if query:

            response = chain.invoke(
                {"input": query},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            st.success(
                response["answer"]
            )


            st.subheader("Sources")
            pages = set()

            for i, doc in enumerate(response["context"]):
                st.write("Page:", doc.metadata.get("page"))
                st.write(doc.page_content[:300])
                st.write("="*50)

            # with st.expander("Retrieved Context"):
            #     for doc in response["context"]:
            #         st.write(
            #             f"Page: {doc.metadata.get('page') + 1}"
            #         )
            #     st.write(doc.page_content)

            #     st.divider()
            with st.expander("Retrieved Context"):
                for i, doc in enumerate(response["context"]):
                    st.write(
                        f"Chunk {i+1} | Page: {doc.metadata.get('page') + 1}"
                    )
                    st.write(doc.page_content)
                    st.divider()