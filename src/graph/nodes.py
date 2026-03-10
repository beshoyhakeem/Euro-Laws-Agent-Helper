from langgraph.graph import StateGraph, END
from src.chains.state import AppState
from src.graph.router import route_selector

from src.chains.chains import  ( classify_question,
                                normal_question,
                                history_question,
                                query_enchance,
                                retrieve_chunks,
                                get_related_celex,
                                retrieve_full_docs,
                                summrize_full_docs_if_needed,
                                rag_answer_chain
                               )


def build_app_graph():
    graph = StateGraph(AppState)

    # Nodes
    graph.add_node("classify", classify_question)
    graph.add_node("normal_answer", normal_question)
    graph.add_node("history_answer", history_question)
    graph.add_node("enhance_query", query_enchance)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("get_related_celex", get_related_celex)
    graph.add_node("retrieve_full_docs", retrieve_full_docs)
    graph.add_node("summrize_full_docs_if_needed", summrize_full_docs_if_needed)
    graph.add_node("rag_answer", rag_answer_chain)

    # Entry
    graph.set_entry_point("classify")

    # Conditional route after classify
    graph.add_conditional_edges(
        "classify",
        route_selector,
        {
            "normal": "normal_answer",
            "rag": "enhance_query",
            "history": "history_answer"
        },
    )

    # RAG branch flow
    graph.add_edge("enhance_query", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "get_related_celex")
    graph.add_edge("get_related_celex", "retrieve_full_docs")
    graph.add_edge("retrieve_full_docs", "summrize_full_docs_if_needed")
    graph.add_edge("summrize_full_docs_if_needed", "rag_answer")

    # End points
    graph.add_edge("normal_answer", END)
    graph.add_edge("history_answer", END)
    graph.add_edge("rag_answer", END)

    return graph.compile()