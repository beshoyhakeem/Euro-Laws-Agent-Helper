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