# terminal_chat.py
from src.graph.graph_builder import build_app_graph
from src.core.resources import get_vectorstore

# --- One-time initialization (load resources + graph once) ---

print("Initializing resources and graph...")

# Force vectorstore initialization once, so embeddings / Weaviate connect a single time
_ = get_vectorstore()

# Build the LangGraph app once
app_graph = build_app_graph()

print("Ready. Start chatting!\n")

# --- Simple terminal chat loop with in-process history ---

def main():
    # history as list of (user, assistant) strings, only used to enrich the question text
    history: list[tuple[str, str]] = []

    print("Euro Laws CLI chat.")
    print("Type 'exit' or 'quit' to stop, or press Ctrl+C.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Bye!")
            break

        # Build a question that includes short conversation history
        # (no changes to your chains or prompts needed)
        if history:
            # Use last 5 turns for context
            recent = history[-5:]
            context_lines = []
            for i, (u, a) in enumerate(recent, 1):
                context_lines.append(f"Turn {i} - User: {u}")
                context_lines.append(f"Turn {i} - Assistant: {a}")
                context_lines.append("")
            context_text = "\n".join(context_lines)

            full_question = (
                "Here is the previous conversation between the user and assistant:\n"
                f"{context_text}\n"
                f"Now the user says: {user_input}\n"
                "Answer the user based on the whole conversation."
            )
        else:
            full_question = user_input

        # Call your existing graph; no internal changes
        result = app_graph.invoke({"question": full_question})
        answer = result["answer"]

        print(f"Assistant: {answer}\n")

        # Save to history for this session
        history.append((user_input, answer))


if __name__ == "__main__":
    main()