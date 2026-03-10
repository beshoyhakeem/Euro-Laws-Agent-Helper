from typing import List, Dict, Any, Literal,TypedDict , Optional
from langchain_core.documents import Document

class AppState(TypedDict, total=False):
    question: str
    route: Literal["normal", "rag", "history"]
    enhanced_query: str
    chunks: str
    relevent_celex: List[str]
    full_docs: str
    docs_list: list[str]
    context: str
    answer: str
    