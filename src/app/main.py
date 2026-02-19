"""
Main application file demonstrating how to use the LangChain/LangGraph integration
with chat history support.
"""
from src.app.langchain_pipeline import RAGAgent
from src.app.langgraph_app import chat, run_app


def example_langchain_usage():
    """Example using LangChain RAGAgent directly."""
    print("=" * 60)
    print("Example: Using LangChain RAGAgent")
    print("=" * 60)
    
    # Create an agent (maintains its own chat history)
    agent = RAGAgent()
    
    # Ask questions
    question1 = "What is the minimum age to work in the EU?"
    answer1 = agent.invoke(question1)
    print(f"\nQ: {question1}")
    print(f"A: {answer1}\n")
    
    # Follow-up question (uses chat history)
    question2 = "What are the exceptions?"
    answer2 = agent.invoke(question2)
    print(f"Q: {question2}")
    print(f"A: {answer2}\n")
    
    # View chat history
    print("Chat History:")
    for msg in agent.get_history():
        print(f"  {msg['role']}: {msg['content'][:100]}...")


def example_langgraph_usage():
    """Example using LangGraph application."""
    print("\n" + "=" * 60)
    print("Example: Using LangGraph Application")
    print("=" * 60)
    
    # Use same thread_id to maintain conversation
    thread_id = "conversation_1"
    
    question1 = "What regulations exist for data protection in the EU?"
    answer1 = chat(question1, thread_id=thread_id)
    print(f"\nQ: {question1}")
    print(f"A: {answer1}\n")
    
    # Follow-up question (conversation persists)
    question2 = "What are the key principles?"
    answer2 = chat(question2, thread_id=thread_id)
    print(f"Q: {question2}")
    print(f"A: {answer2}\n")
    
    # Get full state with all information
    result = run_app("Can you summarize what we discussed?", 
                    config={"configurable": {"thread_id": thread_id}})
    print(f"\nFull State Info:")
    print(f"  Retrieved {len(result['retrieved_docs'])} documents")
    print(f"  Found {len(result['celex_ids'])} unique CELEX IDs")
    print(f"  Answer: {result['answer'][:200]}...")


def example_streaming():
    """Example of streaming execution."""
    print("\n" + "=" * 60)
    print("Example: Streaming Execution")
    print("=" * 60)
    
    from src.app.langgraph_app import run_stream
    
    question = "What are the main EU directives on consumer protection?"
    
    print(f"\nQ: {question}\n")
    print("Streaming execution:")
    
    for event in run_stream(question, config={"configurable": {"thread_id": "stream_1"}}):
        for node_name, node_state in event.items():
            if node_name == "answer" and "answer" in node_state:
                print(f"  [{node_name}] Answer generated: {node_state['answer'][:100]}...")
            elif node_name == "retrieve" and "retrieved_docs" in node_state:
                print(f"  [{node_name}] Retrieved {len(node_state['retrieved_docs'])} documents")


if __name__ == "__main__":
    # Run examples
    try:
        example_langchain_usage()
        example_langgraph_usage()
        example_streaming()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with Azure OpenAI and Weaviate credentials")
        print("2. Installed all requirements: pip install -r requirements.txt")
        print("3. Added langchain-openai and langgraph to requirements.txt")
