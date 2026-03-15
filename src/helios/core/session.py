"""Session management (in-memory for Phase 3, SQLite in Phase 7)."""

from __future__ import annotations

from helios.core.models import Session, SessionStatus


class SessionManager:
    """In-memory session manager."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, objective: str, file_content: str | None = None) -> Session:
        session = Session(objective=objective, file_content=file_content)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def update(self, session: Session) -> None:
        self._sessions[session.id] = session

    def list_all(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def get_latest_incomplete(self) -> Session | None:
        for session in self.list_all():
            if session.status in (SessionStatus.RUNNING, SessionStatus.INTERRUPTED):
                return session
        return None
