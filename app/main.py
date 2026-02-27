from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from src.graph.graph_builder import graph_with_history

app = FastAPI()

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request):
    # Just render the chat page; JS will handle calling /api/chat
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    - req.message: user message
    - req.session_id: client’s session id (string)
    """
    # If client didn’t send a session_id, you can create one (UUID),
    # but simplest is to require the frontend to generate it.
    session_id = req.session_id or "anonymous"

    # Call your LangGraph with session history
    result = graph_with_history.invoke(
        {"question": req.message},
        config={"configurable": {"session_id": session_id}},
    )

    answer = result["answer"]

    return ChatResponse(answer=answer, session_id=session_id)