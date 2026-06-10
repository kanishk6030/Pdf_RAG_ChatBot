# Conversational RAG PDF Chatbot

A Streamlit app that lets you upload one or more PDF files, turns them into chunks, creates embeddings, stores them in a FAISS vector database, and answers questions with a conversational retrieval chain that keeps per-session chat history.

## Flow

![Conversational RAG flow](assets/rag_chatbot.svg)

The flow is:

1. The user enters a Groq API key.
2. The user uploads one or more PDF files.
3. Each PDF is written to a temporary file and loaded with `PyPDFLoader`.
4. The text is split into overlapping chunks with `RecursiveCharacterTextSplitter`.
5. The chunks are converted into embeddings with `HuggingFaceEmbeddings`.
6. The embedded chunks are stored in a FAISS vector store.
7. The vector store is converted into a retriever.
8. A conversational RAG chain is built around the retriever and LLM.
9. The user asks a question.
10. The chain rewrites the question using chat history, retrieves relevant chunks, and generates an answer.
11. The final answer is shown in the Streamlit UI.

## Project Structure

- `app.py` - Streamlit entry point and UI orchestration.
- `config/settings.py` - Environment loading and model / chunk settings.
- `src/pdf_loader.py` - PDF loading and text splitting.
- `src/embeddings.py` - Embedding model setup.
- `src/vector_store.py` - FAISS vector store creation.
- `src/prompts.py` - Prompt templates for question reformulation and answering.
- `src/memory.py` - Session chat history storage.
- `src/rag_chain.py` - Conversational retrieval chain assembly.

## How It Works

### 1. App startup

`app.py` starts the Streamlit interface and asks for a Groq API key. When the key is provided, the app creates a `ChatGroq` LLM instance using the model defined in `config/settings.py`.

### 2. PDF upload and preprocessing

When the user uploads PDFs, each file is saved to a temporary location and passed to `load_and_split()` in `src/pdf_loader.py`.

That function:

- loads the PDF with `PyPDFLoader`
- extracts the document pages
- splits them into chunks with `RecursiveCharacterTextSplitter`
- uses the chunk size and overlap defined in `config/settings.py`

### 3. Embedding generation

`src/embeddings.py` creates a `HuggingFaceEmbeddings` object using the model name from `config/settings.py`.

This turns each chunk into a vector representation that can be searched semantically.

### 4. Vector store creation

`src/vector_store.py` stores all chunk embeddings in a FAISS index.

FAISS is used here because it is fast and works well for local semantic retrieval.

### 5. Conversational retrieval chain

`src/rag_chain.py` builds the final chain in three steps:

- `create_history_aware_retriever()` rewrites the user question using previous chat history.
- `create_stuff_documents_chain()` prepares the answer-generation chain with the QA prompt.
- `create_retrieval_chain()` combines retrieval and generation into one RAG pipeline.

The whole chain is wrapped in `RunnableWithMessageHistory`, which stores chat history per `session_id` using `src/memory.py`.

### 6. Answer generation

When the user asks a question, the app calls `chain.invoke()` with:

- the question as input
- the active `session_id`

The chain then:

- uses the chat history to contextualize the question
- retrieves the most relevant document chunks
- generates an answer from the retrieved context
- returns the response to the UI

## Configuration

The app reads environment variables from `.env`.

Required:

- `HF_TOKEN` - Hugging Face token used by the embedding model.

Optional / internal settings defined in code:

- `CHUNK_SIZE = 2000`
- `CHUNK_OVERLAP = 300`
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`
- `LLM_MODEL = "llama-3.3-70b-versatile"`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
HF_TOKEN=your_hugging_face_token_here
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Enter your Groq API key.
3. Choose or keep the default session ID.
4. Upload one or more PDF files.
5. Ask a question about the uploaded documents.
6. Read the answer shown below the input field.

## Notes

- Multiple PDFs are supported.
- Chat history is kept in Streamlit session state and isolated by `session_id`.
- The app does not persist a long-term vector store between runs; it builds the index from the uploaded PDFs during the current session.
- If no answer is found in the retrieved context, the model is instructed to say it does not know.

## Troubleshooting

- If embeddings fail to load, confirm that `HF_TOKEN` is set correctly in `.env`.
- If the app cannot start, check that your virtual environment is active and dependencies are installed.
- If PDFs appear empty or incomplete, verify that the files are text-based PDFs and not scanned images that require OCR.
