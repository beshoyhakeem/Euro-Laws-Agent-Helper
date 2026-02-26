from src.chains.chains import AppState

def route_selector(state: AppState) -> str:
    """Route to appropriate node based on classification."""
    route = state.get("route")
    
    if route == "normal":
        return "normal"
    elif route == "rag":
        return "rag"
    elif route == "history":
        return "history"

