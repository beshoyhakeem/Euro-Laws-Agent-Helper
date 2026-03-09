from typing import List, Dict, Any, Literal,TypedDict , Optional
from langchain_core.documents import Document

class AppState(TypedDict, total=False):
    question: str
    route: Literal["normal", "rag", "history"]
    enhanced_query: str
    chunks: str
    relevent_celex: List
    full_docs: str
    chunks_for_summrize: list[list[str]]
    summrize_docs: List[str]
    answer: str
    