from src.graph.nodes import build_app_graph

app_graph = build_app_graph()

if __name__ == "__main__":

    result = app_graph.invoke({"question": "Drug dealing Sentences"})
    final_answer = result["answer"]

    print(final_answer)