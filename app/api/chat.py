import json
from fastapi import APIRouter, Depends
from app.api.auth import verify_token
from app.memory.session import SessionStore
from app.memory.conversation import (
    add_message,
    should_summarize,
    summarize_memory
)
from app.vectorstore.faiss_store import FAISSStore
from app.services.retrieval import RetrievalService
from app.graph.workflow import build_graph
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
session_store = SessionStore()
store = FAISSStore(dim=384)
store.load()

retriever = RetrievalService(store)
graph = None

@router.post("/chat")
async def chat(
    query: str,
    session_id: str | None = None,
    user=Depends(verify_token)
    ):
    if not session_id:
        session_id = session_store.create_session()

    memory = session_store.get(session_id)

    add_message(memory, "user", query)

    state = {
        "session_id": session_id,
        "query": query,
        "memory_summary": memory.get("summary", "")
    }

    global graph
    try:
        if graph is None:
            logger.info("Building graph...")
            graph = await build_graph(retriever)
            logger.info("Graph built successfully.")
        result = await graph.ainvoke(state)
        logger.info(f"Graph result: {result}")
    except Exception as e:
        logger.exception("Graph invocation failed")
        return {
            "session_id": session_id,
            "answer": None,
            "source": None,
            "error": str(e)
        }

    add_message(memory, "assistant", result.get("answer", ""))

    if should_summarize(memory):
        memory = summarize_memory(memory)

    session_store.update(session_id, memory)

    return {
        "session_id": session_id,
        "answer": result.get("answer"),
        "source": result.get("source")
    }
