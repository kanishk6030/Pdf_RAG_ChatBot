from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

def get_contextualize_prompt():

    system_prompt = """
    Given a chat history and latest user question,
    reformulate it into a standalone question.
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ]
    )


def get_qa_prompt():

    system_prompt = """
    You are a PDF assistant for question-answering tasks.

    Answer ONLY using the provided context.
    Use the retrieved context to answer.

    If the answer is not contained in the context, respond:
    "I could not find information related to this question in the uploaded documents."

    Do not use external knowledge.

    {context}
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ]
    )