from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from src.prompts import (
    get_contextualize_prompt,
    get_qa_prompt
)

from src.memory import get_session_history


def build_rag_chain(llm,retriever):

    history_aware_retriever = (
        create_history_aware_retriever(
            llm,
            retriever,
            get_contextualize_prompt()
        )
    )

    qa_chain = create_stuff_documents_chain(
        llm,
        get_qa_prompt()
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    conversational_chain = (
        RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer"
        )
    )

    return conversational_chain