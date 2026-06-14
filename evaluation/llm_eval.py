from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json

def evaluate_response(
    question: str,
    context: str,
    answer: str,
    groq_api_key: str
):
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator.

        Question:
        {question}

        Retrieved Context:
        {context}

        Generated Answer:
        {answer}

        Evaluate:

        1. Relevance (1-10)
        2. Faithfulness (1-10)
        3. Completeness (1-10)

        Return ONLY valid JSON:
        Do not use markdown.
            Do not use ```json.
            Do not add explanations outside JSON.


        {{
            "relevance": score,
            "faithfulness": score,
            "completeness": score,
            "overall_feedback": "short feedback"
        }}
        """
        )

    chain = prompt | llm

    result = chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })

    return json.loads(result.content)