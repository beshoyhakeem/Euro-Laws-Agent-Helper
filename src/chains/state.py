from typing import List, Dict, Any, Literal,TypedDict , Optional
from langchain_core.documents import Document

class AppState(TypedDict, total=False):
    question: str
    enhanced_query: str
    docs: List[Document]
    answer: str
    route: Literal["normal", "rag", "history"]