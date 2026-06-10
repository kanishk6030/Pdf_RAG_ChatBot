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
    You are an assistant for question-answering tasks.

    Use the retrieved context to answer.

    If answer is not present,
    say you don't know.

    {context}
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ]
    )