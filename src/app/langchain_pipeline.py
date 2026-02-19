"""
LangChain RAG Pipeline with Chat History Support
"""
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import AzureChatOpenAI
from langchain.memory import ChatMessageHistory

from src.retrieval.retrieval import vectorstore, get_docs_celex, lang_search_chunks
from src.connection.clinets import AZURE_API_KEY, AZURE_ENDPOINT


# Initialize LangChain Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-01-01-preview",
    temperature=0.2,
)

# Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


def format_docs(docs) -> str:
    """Format retrieved documents into a single text block."""
    if not docs:
        return "No documents retrieved."
    
    sections = []
    for i, doc in enumerate(docs, 1):
        celex = doc.metadata.get("celex", "unknown")
        content = doc.page_content
        sections.append(f"<doc {i} celex='{celex}'>\n{content}\n</doc {i}>\n")
    return "\n".join(sections)


# Create the prompt template with chat history support
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant specialized in EU law. 
You help users understand European Union legislation by providing accurate, 
well-sourced answers based on retrieved legal documents.

When answering:
- Use only the relevant parts of the provided legal documents
- Quote article/paragraph numbers when possible
- Reference the CELEX identifier when citing specific documents
- If the documents are not sufficient, say so clearly
- You can reference previous conversation context when relevant"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """User question: {question}

Retrieved documents:
{context}

Please provide a comprehensive answer based on the documents above and any relevant context from our conversation."""),
])


def create_rag_chain(memory: ChatMessageHistory):
    """
    Create a RAG chain with chat history support.
    
    Args:
        memory: ChatMessageHistory instance to maintain conversation state
        
    Returns:
        Runnable chain that processes questions with context
    """
    def retrieve_context(question: str) -> Dict[str, Any]:
        """Retrieve relevant documents for the question."""
        docs = retriever.invoke(question)
        return {
            "question": question,
            "context": format_docs(docs),
            "docs": docs
        }
    
    def format_chat_history() -> List[BaseMessage]:
        """Format chat history for the prompt."""
        return memory.messages
    
    chain = (
        {
            "question": RunnablePassthrough(),
            "chat_history": lambda x: format_chat_history(),
            "context": lambda x: retrieve_context(x["question"])["context"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


class RAGAgent:
    """
    RAG Agent with chat history management.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize RAG Agent with memory.
        
        Args:
            session_id: Optional session identifier for persistent memory
        """
        self.memory = ChatMessageHistory()
        self.session_id = session_id
        self.chain = create_rag_chain(self.memory)
    
    def invoke(self, question: str) -> str:
        """
        Process a question and return the answer.
        
        Args:
            question: User's question
            
        Returns:
            Assistant's answer
        """
        # Get answer from chain
        answer = self.chain.invoke({"question": question})
        
        # Add to memory
        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)
        
        return answer
    
    def get_full_docs(self, question: str) -> List[str]:
        """
        Get full documents for a question using celex IDs.
        
        Args:
            question: User's question
            
        Returns:
            List of full document texts
        """
        docs, celex_ids = lang_search_chunks(question)
        return get_docs_celex(celex_ids)
    
    def clear_history(self):
        """Clear chat history."""
        self.memory.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get chat history as a list of dictionaries.
        
        Returns:
            List of messages with 'role' and 'content' keys
        """
        history = []
        for msg in self.memory.messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history


# Convenience function for simple usage
def answer_question(question: str, agent: Optional[RAGAgent] = None) -> str:
    """
    Simple function to answer a question with optional agent instance.
    
    Args:
        question: User's question
        agent: Optional RAGAgent instance (creates new one if not provided)
        
    Returns:
        Answer string
    """
    if agent is None:
        agent = RAGAgent()
    return agent.invoke(question)
