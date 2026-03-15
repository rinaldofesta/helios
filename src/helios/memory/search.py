"""Session search via FTS5."""

from __future__ import annotations

from helios.core.models import Session
from helios.memory.store import SessionStore


class SessionSearch:
    """Full-text search over session history."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def search(self, query: str, limit: int = 10) -> list[Session]:
        """Search sessions by objective and exchange content."""
        return await self._store.search(query, limit)
