from typing import List, Dict, Any, Literal,TypedDict , D , Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.connection.clinets import llm , weaviate_client
from src.prompts.rag_prompts import *
from src.retrieval.retrieval import lang_search_chunks


class AppState(TypedDict, total=False):
    question: str
    enhanced_query: str
    docs: List[Document]
    answer: str
    route: Literal["normal", "rag"]

# Chain to Classify User Question
def classify_question(state: AppState) -> AppState:

    chain = classifier_prompt | llm | StrOutputParser()
    route_raw: str = chain.invoke({"question": state["question"]}).strip().lower()

    # Normalise LLM output to exactly "normal" or "rag"
    if "rag" in route_raw:
        route: Literal["normal", "rag"] = "rag"
    else:
        route = "normal"

    return {"route": route}
