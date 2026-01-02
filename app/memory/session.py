from typing import Dict
from uuid import uuid4


class SessionStore:
    """
    Simple in-memory session store.
    Replace with Redis/Postgres later.
    """

    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = {
            "history": [],
            "summary": ""
        }
        return session_id

    def get(self, session_id: str) -> dict:
        return self.sessions.setdefault(
            session_id,
            {"history": [], "summary": ""}
        )

    def update(self, session_id: str, data: dict):
        self.sessions[session_id] = data
