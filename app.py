import html
import os
import tempfile

import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq

from config.settings import LLM_MODEL
from evaluation.llm_eval import evaluate_response
from evaluation.retrieval_eval import evaluate_retriever
from src.embeddings import get_embeddings
from src.pdf_loader import load_and_split
from src.rag_chain import build_rag_chain
from src.retrievers.hybrid_retriever import get_hybrid_retriever
from src.vector_store import create_vectorstore


st.set_page_config(
    page_title="Conversational RAG Chatbot",
    page_icon="PDF",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --surface: #ffffff;
        --surface-soft: #f6f7f9;
        --app-bg: #f5f7fa;
        --ink: #18212f;
        --accent: #0b6b60;
        --muted: #667085;
        --line: #d8dee8;
        --header-bg: rgba(245, 247, 250, 0.92);
        --dock-bg: rgba(245, 247, 250, 0.94);
        --sidebar-bg: #ffffff;
        --sidebar-text: #18212f;
        --sidebar-muted: #667085;
        --sidebar-field-bg: #f6f7f9;
        --sidebar-field-border: #d8dee8;
        --field-bg: #ffffff;
        --field-text: #18212f;
        --field-placeholder: #8a95a6;
        --field-border: #d8dee8;
        --accent-strong: #084f49;
        --accent-soft: #e6f3f0;
        --warm: #b45309;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --surface: #151b23;
            --surface-soft: #1c2430;
            --app-bg: #0b1118;
            --ink: #edf3f8;
            --accent: #2dd4bf;
            --muted: #a8b3c2;
            --line: #2a3544;
            --header-bg: rgba(11, 17, 24, 0.92);
            --dock-bg: rgba(11, 17, 24, 0.94);
            --sidebar-bg: #090f16;
            --sidebar-text: #edf3f8;
            --sidebar-muted: rgba(237, 243, 248, 0.68);
            --sidebar-field-bg: #151b23;
            --sidebar-field-border: #2a3544;
            --field-bg: #151b23;
            --field-text: #edf3f8;
            --field-placeholder: #a8b3c2;
            --field-border: #2a3544;
            --accent-strong: #5eead4;
            --accent-soft: rgba(45, 212, 191, 0.12);
            --warm: #f59e0b;
        }
    }

    html[data-theme="light"],
    body[data-theme="light"],
    [data-theme="light"],
    [data-baseweb-theme="light"] {
        --surface: #ffffff;
        --surface-soft: #f6f7f9;
        --app-bg: #f5f7fa;
        --ink: #18212f;
        --accent: #0b6b60;
        --muted: #667085;
        --line: #d8dee8;
        --header-bg: rgba(245, 247, 250, 0.92);
        --dock-bg: rgba(245, 247, 250, 0.94);
        --sidebar-bg: #ffffff;
        --sidebar-text: #18212f;
        --sidebar-muted: #667085;
        --sidebar-field-bg: #f6f7f9;
        --sidebar-field-border: #d8dee8;
        --field-bg: #ffffff;
        --field-text: #18212f;
        --field-placeholder: #8a95a6;
        --field-border: #d8dee8;
        --accent-strong: #084f49;
        --accent-soft: #e6f3f0;
        --warm: #b45309;
    }

    html[data-theme="dark"],
    body[data-theme="dark"],
    [data-theme="dark"],
    [data-baseweb-theme="dark"] {
        --surface: #151b23;
        --surface-soft: #1c2430;
        --app-bg: #0b1118;
        --ink: #edf3f8;
        --accent: #2dd4bf;
        --muted: #a8b3c2;
        --line: #2a3544;
        --header-bg: rgba(11, 17, 24, 0.92);
        --dock-bg: rgba(11, 17, 24, 0.94);
        --sidebar-bg: #090f16;
        --sidebar-text: #edf3f8;
        --sidebar-muted: rgba(237, 243, 248, 0.68);
        --sidebar-field-bg: #151b23;
        --sidebar-field-border: #2a3544;
        --field-bg: #151b23;
        --field-text: #edf3f8;
        --field-placeholder: #a8b3c2;
        --field-border: #2a3544;
        --accent-strong: #5eead4;
        --accent-soft: rgba(45, 212, 191, 0.12);
        --warm: #f59e0b;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background: var(--app-bg);
        color: var(--ink);
    }

    .stApp p,
    .stApp span,
    .stApp label,
    .stMarkdown,
    [data-testid="stCaptionContainer"] {
        color: inherit;
    }

    .stApp input,
    .stApp textarea {
        color: var(--field-text);
        -webkit-text-fill-color: var(--field-text);
    }

    .stTextInput div[data-baseweb="input"],
    .stTextArea div[data-baseweb="textarea"],
    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        background: var(--field-bg);
        border-color: var(--field-border);
        color: var(--field-text);
    }

    .stTextInput input,
    .stTextArea textarea,
    input,
    textarea {
        background: var(--field-bg);
        color: var(--field-text);
        -webkit-text-fill-color: var(--field-text);
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    input::placeholder,
    textarea::placeholder {
        color: var(--field-placeholder);
        opacity: 1;
    }

    header[data-testid="stHeader"] {
        background: var(--header-bg);
        border-bottom: 1px solid var(--line);
        backdrop-filter: blur(10px);
    }

    .block-container {
        max-width: 1160px;
        padding: 3rem 1.5rem 4.6rem;
    }

    [data-testid="column"] > div {
        height: 100%;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--line);
        box-shadow: 8px 0 24px color-mix(in srgb, var(--ink) 8%, transparent);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.35rem 1rem 1.6rem;
    }

    [data-testid="stSidebar"] * {
        color: var(--sidebar-text);
    }

    [data-testid="stSidebar"] label {
        color: var(--sidebar-text) !important;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] textarea {
        min-height: 2.55rem;
        background: var(--sidebar-field-bg);
        border: 1px solid var(--sidebar-field-border);
        border-radius: 8px;
        color: var(--sidebar-text);
        -webkit-text-fill-color: var(--sidebar-text);
    }

    [data-testid="stSidebar"] div[data-baseweb="input"],
    [data-testid="stSidebar"] div[data-baseweb="textarea"] {
        background: var(--sidebar-field-bg);
        border-color: var(--sidebar-field-border);
        color: var(--sidebar-text);
    }

    [data-testid="stSidebar"] .stTextInput input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: var(--sidebar-muted);
        -webkit-text-fill-color: var(--sidebar-muted);
        opacity: 1;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.72rem;
    }

    [data-testid="stSidebar"] .stFileUploader section {
        background: var(--sidebar-field-bg);
        border: 1px dashed var(--sidebar-field-border);
        border-radius: 8px;
        padding: 0.85rem;
    }

    [data-testid="stSidebar"] .stFileUploader section div {
        font-size: 0.88rem;
    }

    [data-testid="stSidebar"] .stFileUploader button {
        background: var(--field-bg);
        border: 1px solid var(--field-border);
        color: var(--field-text);
    }

    [data-testid="stSidebar"] svg,
    [data-testid="stChatInput"] svg {
        color: var(--field-placeholder);
        fill: currentColor;
        stroke: currentColor;
    }

    .hero-panel,
    .status-panel,
    .glass-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 5%, transparent);
    }

    .hero-panel {
        min-height: 198px;
        padding: 1.25rem 1.35rem;
        position: relative;
        overflow: hidden;
    }

    .hero-panel::after {
        background: linear-gradient(180deg, rgba(11, 107, 96, 0.14), rgba(11, 107, 96, 0.02));
        border-left: 1px solid rgba(11, 107, 96, 0.18);
        content: "";
        height: 100%;
        position: absolute;
        right: 0;
        top: 0;
        width: 11px;
    }

    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.05rem;
    }

    .hero-pill {
        background: var(--surface-soft);
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--ink);
        font-size: 0.8rem;
        font-weight: 700;
        line-height: 1;
        padding: 0.48rem 0.7rem;
    }

    .eyebrow {
        color: var(--accent-strong);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.42rem;
    }

    .hero-panel h1 {
        color: var(--ink);
        font-size: clamp(1.95rem, 3vw, 3.05rem);
        line-height: 1.04;
        letter-spacing: 0;
        margin: 0 0 0.65rem;
        max-width: 760px;
    }

    .hero-panel p {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.58;
        max-width: 700px;
        margin: 0;
    }

    .status-panel {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 198px;
        padding: 1rem;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        gap: 0.7rem;
        border-bottom: 1px solid var(--line);
        padding: 0.72rem 0;
    }

    .status-row:first-child {
        padding-top: 0;
    }

    .status-row:last-child {
        border-bottom: 0;
        padding-bottom: 0;
    }

    .status-label {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
    }

    .status-value {
        color: var(--ink);
        font-weight: 800;
        text-align: right;
    }

    .sidebar-brand {
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1rem;
        padding-bottom: 0.95rem;
    }

    .sidebar-brand h2 {
        font-size: 1.22rem;
        line-height: 1.2;
        margin: 0 0 0.28rem;
    }

    .sidebar-brand p,
    .sidebar-note {
        color: var(--sidebar-muted);
        font-size: 0.86rem;
        line-height: 1.45;
        margin: 0;
    }

    .file-chip {
        background: var(--sidebar-field-bg);
        border: 1px solid var(--sidebar-field-border);
        border-radius: 8px;
        margin: 0.3rem 0;
        padding: 0.5rem 0.65rem;
        font-size: 0.84rem;
        overflow-wrap: anywhere;
    }

    .empty-state {
        background: var(--surface);
        border: 1px dashed var(--line);
        border-radius: 8px;
        color: var(--muted);
        margin: 0.95rem 0 1rem;
        padding: 0.95rem 1rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 800;
        margin: 1rem 0 0.5rem;
    }

    .stButton > button {
        background: var(--accent);
        border: 1px solid var(--accent);
        border-radius: 8px;
        color: white;
        font-weight: 700;
        min-height: 2.55rem;
        width: 100%;
    }

    .stButton > button:hover {
        background: var(--accent-strong);
        border-color: var(--accent-strong);
        color: white;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 5%, transparent);
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 5%, transparent);
        margin: 0.55rem 0;
        padding: 0.82rem 0.95rem;
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.58;
        margin-bottom: 0.35rem;
    }

    [data-testid="stChatInput"] {
        background: var(--dock-bg);
        border: 0;
        bottom: 0;
        left: 21rem;
        padding: 0.45rem 1.5rem 0.65rem;
        position: fixed;
        right: 0;
        backdrop-filter: blur(10px);
        box-shadow: none;
        z-index: 999;
    }

    [data-testid="stChatInput"] > div {
        max-width: 1160px;
        margin: 0 auto;
        width: 100%;
    }

    [data-testid="stChatInput"] > div > div {
        width: 100%;
    }

    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] label {
        width: 100%;
    }

    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background: var(--field-bg);
        border: 0;
        border-radius: 8px;
        box-shadow: none;
        min-height: 2.45rem;
        outline: 0;
        width: 100%;
    }

    [data-testid="stChatInput"] div[data-baseweb="textarea"] *,
    [data-testid="stChatInput"] div[data-baseweb="base-input"] * {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: 0 !important;
    }

    [data-testid="stChatInput"] textarea {
        background: var(--field-bg);
        border: 0;
        border-radius: 8px;
        box-shadow: none;
        color: var(--field-text);
        font-size: 0.94rem;
        line-height: 1.45;
        min-height: 2.45rem;
        padding: 0.58rem 3rem 0.58rem 0.8rem;
        -webkit-text-fill-color: var(--field-text);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--field-placeholder);
        opacity: 1;
        -webkit-text-fill-color: var(--field-placeholder);
    }

    [data-testid="stChatInput"] textarea:focus {
        border: 0;
        box-shadow: none;
        outline: none;
    }

    [data-testid="stChatInput"] textarea:disabled {
        background: var(--field-bg);
        color: var(--field-placeholder);
        opacity: 1;
        -webkit-text-fill-color: var(--field-placeholder);
    }

    [data-testid="stChatInput"] button {
        background: var(--accent);
        border-radius: 8px;
        color: #ffffff;
        height: 2.2rem;
        margin-right: 0.1rem;
        width: 2.2rem;
    }

    [data-testid="stChatInput"] button:disabled {
        background: color-mix(in srgb, var(--ink) 8%, var(--surface));
        color: var(--muted);
        opacity: 1;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 700;
        padding: 0.55rem 0.85rem;
    }

    .streamlit-expanderHeader {
        font-weight: 700;
    }

    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 5%, transparent);
        margin-bottom: 0.55rem;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 2rem;
        }

        .hero-panel {
            min-height: auto;
            padding: 1.05rem;
        }

        .status-panel {
            min-height: auto;
        }

        [data-testid="stChatInput"] {
            left: 0;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": "Hi. Upload a few PDFs, build the knowledge base, and I will answer from the retrieved context.",
            }
        ],
        "rag_chain": None,
        "processed_files": [],
        "last_context": [],
        "last_evaluation": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_knowledge_base(uploaded_files, api_key):
    documents = []
    temp_paths = []

## Uploaded PDFs are temporarily saved to disk, loaded, and split into chunks. After processing, the temporary files are deleted.
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            temp_paths.append(tmp.name)

        documents.extend(load_and_split(temp_paths[-1]))

## this is to delete the temporary files after processing. It is important to clean up temporary files to avoid clutter and potential security risks.
    for path in temp_paths:
        try:
            os.unlink(path)
        except OSError:
            pass

## Embeddings are generated for the documents, a vector store is created, and a hybrid retriever is set up. The retriever is evaluated to ensure it works correctly. Finally, a RAG chain is built using the LLM and the retriever, and both the chain and the documents are returned.
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(documents, embeddings)
    retriever = get_hybrid_retriever(documents, vectorstore)
    # evaluate_retriever(retriever)

    llm = ChatGroq(api_key=api_key, model=LLM_MODEL)
    chain = build_rag_chain(llm, retriever)

    return chain, documents

## This  function renders the context tabs in the Streamlit app. It takes the response from the RAG chain and displays the sources and retrieved chunks in separate tabs. Each source and chunk is displayed in an expandable section, allowing users to view the content of each document or chunk. The first source is expanded by default for better visibility.
def render_context_tabs(response):
    context = response.get("context", [])
    if not context:
        return

    st.markdown('<div class="section-title">Evidence</div>', unsafe_allow_html=True)
    sources_tab, chunks_tab = st.tabs(["Sources", "Retrieved Chunks"])

    with sources_tab:
        for index, doc in enumerate(context, start=1):
            page = doc.metadata.get("page")
            page_label = page + 1 if isinstance(page, int) else "Unknown"
            with st.expander(f"Source {index} - Page {page_label}", expanded=index == 1):
                st.write(doc.page_content[:700])

    with chunks_tab:
        for index, doc in enumerate(context, start=1):
            page = doc.metadata.get("page")
            page_label = page + 1 if isinstance(page, int) else "Unknown"
            with st.expander(f"Chunk {index} - Page {page_label}", expanded=False):
                st.write(doc.page_content)


#This function evaluates the quality of the answer provided by the RAG chain. It takes the response and the API key as inputs, extracts the context from the response, and calls the evaluate_response function to get an evaluation of relevance, faithfulness, and completeness. The evaluation results are displayed in a metrics format, and if there is any overall feedback, it is shown as a caption. The evaluation results are also saved to a CSV file for record-keeping.
def render_evaluation(response, api_key):
    context_text = "\n\n".join(doc.page_content for doc in response.get("context", []))

    evaluation = evaluate_response(
        question=response["input"],
        context=context_text,
        answer=response["answer"],
        groq_api_key=api_key,
    )

    st.session_state.last_evaluation = evaluation

    st.markdown('<div class="section-title">Answer Quality</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Relevance", value=evaluation["relevance"])
    with col2:
        st.metric(label="Faithfulness", value=evaluation["faithfulness"])
    with col3:
        st.metric(label="Completeness", value=evaluation["completeness"])

    if evaluation.get("overall_feedback"):
        st.caption(evaluation["overall_feedback"])

    row = {
        "question": response["input"],
        "answer": response["answer"],
        "relevance": evaluation["relevance"],
        "faithfulness": evaluation["faithfulness"],
        "completeness": evaluation["completeness"],
        "feedback": evaluation["overall_feedback"],
    }

    pd.DataFrame([row]).to_csv(
        "evaluation_results.csv",
        mode="a",
        header=False,
        index=False,
    )


init_state()


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>Document Studio</h2>
            <p>Private, source-grounded answers over your uploaded PDFs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key = st.text_input("Groq API Key", type="password")
    session_id = st.text_input("Session ID", value="default")
    uploaded_files = st.file_uploader(
        "PDF Library",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown('<p class="sidebar-note">Queued files</p>', unsafe_allow_html=True)
        for file in uploaded_files:
            safe_name = html.escape(file.name)
            st.markdown(f'<div class="file-chip">{safe_name}</div>', unsafe_allow_html=True)

    process_disabled = not api_key or not uploaded_files
    process = st.button("Build Knowledge Base", disabled=process_disabled)

    if process:
        with st.spinner("Reading PDFs, creating embeddings, and preparing retrieval..."):
            chain, documents = build_knowledge_base(uploaded_files, api_key)
            st.session_state.rag_chain = chain
            st.session_state.processed_files = [file.name for file in uploaded_files]
            st.session_state.document_count = len(documents)
        st.success("Knowledge base ready.")

    if process_disabled:
        st.caption("Add a Groq API key and at least one PDF to begin.")


ready = st.session_state.rag_chain is not None
file_count = len(st.session_state.processed_files)
document_count = st.session_state.get("document_count", 0)

hero_col, status_col = st.columns([3.25, 1.1], gap="small")

with hero_col:
    st.markdown(
        """
        <section class="hero-panel">
            <div class="eyebrow">Conversational RAG workspace</div>
            <h1>Review PDFs with traceable answers.</h1>
            <div class="hero-meta">
                <span class="hero-pill">Hybrid retrieval</span>
                <span class="hero-pill">Page evidence</span>
                <span class="hero-pill">LLM evaluation</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

with status_col:
    st.markdown(
        f"""
        <aside class="status-panel">
            <div class="status-row">
                <span class="status-label">Status</span>
                <span class="status-value">{'Ready' if ready else 'Setup needed'}</span>
            </div>
            <div class="status-row">
                <span class="status-label">PDFs</span>
                <span class="status-value">{file_count}</span>
            </div>
            <div class="status-row">
                <span class="status-label">Chunks</span>
                <span class="status-value">{document_count}</span>
            </div>
        </aside>
        """,
        unsafe_allow_html=True,
    )

if not ready:
    st.markdown(
        """
        <div class="empty-state">
            Start in the sidebar: add your Groq key, upload PDFs, and build the knowledge base.
            The chat will unlock as soon as retrieval is ready.
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input(
    "Ask a question about your documents",
    disabled=not ready,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Finding the strongest supporting context..."):
            response = st.session_state.rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}},
            )
            st.markdown(response["answer"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
        }
    )
    st.session_state.last_context = response.get("context", [])

    # render_context_tabs(response)
    # render_evaluation(response, api_key)
