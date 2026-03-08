from langchain_core.output_parsers import StrOutputParser ,JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.chains.state import AppState, Literal
from src.generation.chat import llm , llm_reason
from src.prompts.rag_prompts import *
from src.retrieval.retrieval import get_full_docs_celex, search_docs_hybrid


# Chain to Classify User Question
def classify_question(state: AppState) -> AppState:

    chain = classifier_prompt | llm_reason | StrOutputParser()
    route_raw: str = chain.invoke({"question": state["question"]}).strip().lower()

    # Normalise LLM output to exactly "normal" or "rag"
    if "rag" in route_raw:
        route: Literal["normal", "rag", "history"] = "rag"
    elif "history" in route_raw:
        route: Literal["normal", "rag", "history"] = "history"
    else:
        route: Literal["normal", "rag", "history"] = "normal"
    
    # For debugging purpuse
    print(f"route : {route}")

    return {"route": route}

# Chain to answer normal User Question
def normal_question(state: AppState) -> AppState:

    chain = normal_answer_prompt| llm | StrOutputParser()
    normal_que_answer: str = chain.invoke({"question": state["question"]})

    return {"answer": normal_que_answer}

# Chain to answer User Question dependinmg on history
def history_question(state: AppState) -> AppState:

    chain = normal_answer_prompt| llm | StrOutputParser()
    history_que_answer: str = chain.invoke({"question": state["question"]})
    return {"answer": history_que_answer}


# Chain to enhance User Question for RAG
def query_enchance(state: AppState) -> AppState:

    chain = query_enhance_prompt | llm | StrOutputParser()
    enhaced_query = chain.invoke({"question": state["question"]})

    # For debugging purpuse
    print(f"enhaced_query : {enhaced_query}")

    return {"enhanced_query": enhaced_query }

# function to retrive chunks
def retrieve_chunks(state: AppState) -> AppState:

    query : str = state.get("enhanced_query")
    chunks = search_docs_hybrid(query)

    # For debugging purpuse
    #print(f"chunks :\n {chunks}")
    return {"chunks": chunks}

# function to get the celex of releted chunks
def get_related_celex(state: AppState) -> AppState:

    chain = check_relevant_chunks_prompt | llm | JsonOutputParser()
    relevent_celex = chain.invoke(
        {
            "question": state["question"],
            "chunks": state["chunks"],
        }
    )

    print(f"relevent_celex: {relevent_celex}")

    print(type(relevent_celex), relevent_celex)

    return {"relevent_celex": relevent_celex}

# function to summrize docs
def summrize_full_docs(state: AppState) -> AppState:

    return     

# function to retrive docs
def retrieve_full_docs(state: AppState) -> AppState:

    celex_ids : list = state.get("relevent_celex")
    docs = get_full_docs_celex(celex_ids)

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

