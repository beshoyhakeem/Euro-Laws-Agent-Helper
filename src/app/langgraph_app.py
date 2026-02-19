"""
LangGraph Application with Chat History Support
"""
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.app.langchain_pipeline import RAGAgent, format_docs
from src.retrieval.retrieval import get_docs_celex, lang_search_chunks
from src.connection.clinets import weaviate_client
from langchain_weaviate import WeaviateVectorStore
from src.sentencetransformers.st_class import SentenceTransformersEmbeddings


# Initialize components
embedding_model = SentenceTransformersEmbeddings('sentence-transformers/all-mpnet-base-v2')
vectorstore = WeaviateVectorStore(
    client=weaviate_client,
    index_name="Euro_Laws",
    text_key="text",
    embedding=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


class GraphState(TypedDict):
    """State for the LangGraph application."""
    question: str
    chat_history: Annotated[List[BaseMessage], "append"]
    retrieved_docs: List
    celex_ids: List[str]
    full_docs: List[str]
    answer: str
    rag_agent: RAGAgent


def retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieve relevant documents for the question.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with retrieved documents
    """
    question = state["question"]
    
    # Retrieve chunks and get celex IDs
    docs, celex_ids = lang_search_chunks(question)
    
    # Get full documents
    full_docs = get_docs_celex(celex_ids)
    
    return {
        "retrieved_docs": docs,
        "celex_ids": celex_ids,
        "full_docs": full_docs,
    }


def answer_node(state: GraphState) -> GraphState:
    """
    Generate answer using RAG agent with chat history.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with answer
    """
    question = state["question"]
    
    # Get or create RAG agent
    if "rag_agent" not in state or state["rag_agent"] is None:
        agent = RAGAgent()
    else:
        agent = state["rag_agent"]
    
    # Get answer with chat history
    answer = agent.invoke(question)
    
    return {
        "answer": answer,
        "rag_agent": agent,
        "chat_history": agent.memory.messages,
    }


def format_context_node(state: GraphState) -> GraphState:
    """
    Format context for display/logging (optional node).
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state (no changes, just for logging)
    """
    # This node can be used for logging or formatting
    # For now, just pass through
    return {}


# Build the graph
def create_graph() -> StateGraph:
    """
    Create and compile the LangGraph application.
    
    Returns:
        Compiled graph application
    """
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("format_context", format_context_node)
    
    # Define the flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "format_context")
    graph.add_edge("format_context", END)
    
    # Add memory for checkpointing (enables conversation persistence)
    memory = MemorySaver()
    
    return graph.compile(checkpointer=memory)


# Create the app instance
app = create_graph()


def run_app(question: str, config: dict = None) -> dict:
    """
    Run the LangGraph application with a question.
    
    Args:
        question: User's question
        config: Optional configuration dict with thread_id for conversation persistence
        
    Returns:
        Final state dictionary with answer and other information
    """
    if config is None:
        config = {"configurable": {"thread_id": "default"}}
    
    initial_state = {
        "question": question,
        "chat_history": [],
        "retrieved_docs": [],
        "celex_ids": [],
        "full_docs": [],
        "answer": "",
        "rag_agent": None,
    }
    
    final_state = app.invoke(initial_state, config)
    return final_state


def run_stream(question: str, config: dict = None):
    """
    Stream the execution of the graph (for real-time updates).
    
    Args:
        question: User's question
        config: Optional configuration dict with thread_id
        
    Yields:
        State updates as the graph executes
    """
    if config is None:
        config = {"configurable": {"thread_id": "default"}}
    
    initial_state = {
        "question": question,
        "chat_history": [],
        "retrieved_docs": [],
        "celex_ids": [],
        "full_docs": [],
        "answer": "",
        "rag_agent": None,
    }
    
    for event in app.stream(initial_state, config):
        yield event


# Convenience function for simple usage
def chat(question: str, thread_id: str = "default") -> str:
    """
    Simple chat function with conversation persistence.
    
    Args:
        question: User's question
        thread_id: Conversation thread ID (use same ID for same conversation)
        
    Returns:
        Answer string
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = run_app(question, config)
    return result["answer"]
