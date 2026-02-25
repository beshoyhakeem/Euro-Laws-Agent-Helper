from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.chains.state import AppState, Literal
from src.generation.chat import llm , llm_reason
from src.prompts.rag_prompts import *
from src.retrieval.retrieval import search_docs


# Chain to Classify User Question
def classify_question(state: AppState) -> AppState:

    chain = classifier_prompt | llm_reason | StrOutputParser()
    route_raw: str = chain.invoke({"question": state["question"]}).strip().lower()

    # Normalise LLM output to exactly "normal" or "rag"
    if "rag" in route_raw:
        route: Literal["normal", "rag"] = "rag"
    else:
        route = "normal"
    
    # For debugging purpuse
    print(f"route : {route}")

    return {"route": route}

# Chain to answer normal User Question
def normal_question(state: AppState) -> AppState:

    chain = normal_answer_prompt| llm | StrOutputParser()
    normal_que_answer: str = chain.invoke({"question": state["question"]})

    return {"answer": normal_que_answer}

# Chain to enhance User Question for RAG
def query_enchance(state: AppState) -> AppState:

    chain = query_enhance_prompt | llm | StrOutputParser()
    enhaced_query = chain.invoke({"question": state["question"]})

    # For debugging purpuse
    print(f"enhaced_query : {enhaced_query}")

    return {"enhanced_query": enhaced_query }

# function to retrive docs
def retrieve_docs(state: AppState) -> AppState:

    query : str = state.get("enhanced_query")
    docs = search_docs(query)

    # For debugging purpuse
    #print(f"docs :\n {docs}")
    
    return {"docs": docs}

# Chain to answer User Question form docs "RAG"
def rag_answer_chain(state: AppState) -> AppState:

    chain = rag_answer_prompt | llm | StrOutputParser()
    rag_answer = chain.invoke(
        {
            "question": state["question"],
            "context": state["docs"],
        }
    )

    return {"answer" : rag_answer}

if __name__ == "__main__":

    print ("true")

    state: AppState = {"question": "hello can i know what is sun"}
    print(classify_question(state))

