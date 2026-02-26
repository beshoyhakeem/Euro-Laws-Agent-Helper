from src.graph.nodes import build_app_graph
from src.graph.history import RunnableWithMessageHistory, get_session_history 

app_graph = build_app_graph()

graph_with_history = RunnableWithMessageHistory(
    app_graph,
    get_session_history=get_session_history,
    input_messages_key="question",  # your user input key in AppState
    output_messages_key="answer",   # your final answer key in AppState
)

if __name__ == "__main__":


    q = """"
    Hello everyone, my divorce was finalized in 2024. And my ex is selling her property which we split during divorce. But we both are on deed till date when selling. There’s an offer in place. The property is being sold for profit close to 250K. And all proceeds will be going to my Ex. I have been told I have to be there for signing paperwork next week. 
    1). How do I make sure the buyer doesn’t sue me in the future after buying the property for any kind of reasons. 
    2). How do I make sure my ex is responsible for capital gains I’m not held responsible for those gains to pay tax.
    3). Thankyou 🙏🏾
    
    """
    """
    result = app_graph.invoke({"question": q})
    final_answer = result["answer"]
    
    print(final_answer)
    """

    """
    session_id = ""

    result = graph_with_history.invoke(
    {"question": q},
    config={"configurable": {"session_id": session_id}},
    )
    answer = result["answer"]
    """

    session_id = "user-1"  # or from user auth / tab id

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        result = graph_with_history.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print("Assistant:", result["answer"])