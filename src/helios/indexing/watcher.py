"""File watcher — monitors indexed directories and triggers re-indexing on changes."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Coroutine

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 2.0


class _ChangeHandler(FileSystemEventHandler):
    """Collects file system events and notifies the watcher."""

    def __init__(self, source_id: str, notify: Callable[[str], None]) -> None:
        self.source_id = source_id
        self._notify = notify

    def on_any_event(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = getattr(event, "src_path", "")
        # Skip noise
        if any(seg in path for seg in ("/.git/", "/__pycache__/", "/node_modules/", "/.venv/")):
            return
        self._notify(self.source_id)


class FileWatcher:
    """Watches directories for changes and triggers async re-indexing with debouncing.

    Usage:
        watcher = FileWatcher(on_change=my_async_reindex_function)
        watcher.watch(source_id, "/path/to/project")
        await watcher.start()
        ...
        await watcher.stop()
    """

    def __init__(
        self,
        on_change: Callable[[str], Coroutine],
        debounce: float = DEBOUNCE_SECONDS,
    ) -> None:
        self._on_change = on_change
        self._debounce = debounce
        self._observer = Observer()
        self._observer.daemon = True
        self._pending: dict[str, float] = {}  # source_id -> last_change_time
        self._task: asyncio.Task | None = None
        self._running = False
        self._watched: set[str] = set()

    def watch(self, source_id: str, path: str) -> None:
        """Start watching a directory for changes."""
        if path in self._watched:
            return
        if not Path(path).is_dir():
            return
        handler = _ChangeHandler(source_id, self._mark_changed)
        self._observer.schedule(handler, path, recursive=True)
        self._watched.add(path)
        logger.debug("Watching: %s", path)

    def _mark_changed(self, source_id: str) -> None:
        self._pending[source_id] = time.time()

    async def start(self) -> None:
        """Start the file watcher and background processing loop."""
        if self._running:
            return
        self._observer.start()
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("File watcher started (%d directories)", len(self._watched))

    async def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        self._observer.stop()
        self._observer.join(timeout=5)
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("File watcher stopped")

    async def _process_loop(self) -> None:
        """Background loop that processes debounced changes."""
        while self._running:
            await asyncio.sleep(1)
            now = time.time()
            ready = [
                sid for sid, ts in self._pending.items()
                if now - ts >= self._debounce
            ]
            for sid in ready:
                del self._pending[sid]
                try:
                    logger.info("Auto-reindexing source: %s", sid)
                    await self._on_change(sid)
                except Exception:
                    logger.warning("Auto-reindex failed for %s", sid, exc_info=True)
