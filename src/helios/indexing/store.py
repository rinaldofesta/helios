"""Content indexing store — SQLite + FTS5 for indexed documents."""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL DEFAULT 'directory',
    path TEXT NOT NULL,
    file_count INTEGER DEFAULT 0,
    total_size INTEGER DEFAULT 0,
    indexed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    path TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    content TEXT NOT NULL,
    language TEXT DEFAULT '',
    size INTEGER DEFAULT 0,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(id)
);

CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    document_id UNINDEXED,
    relative_path,
    content
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    start_line INTEGER DEFAULT 0,
    end_line INTEGER DEFAULT 0,
    embedding BLOB,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content
);
"""


# --------------------------------------------------------------------------- #
#  Models
# --------------------------------------------------------------------------- #


class Source(BaseModel):
    id: str
    name: str
    source_type: str = "directory"
    path: str
    file_count: int = 0
    total_size: int = 0
    indexed_at: datetime
    updated_at: datetime


class Document(BaseModel):
    id: str
    source_id: str
    path: str
    relative_path: str
    content: str
    language: str = ""
    size: int = 0
    content_hash: str
    indexed_at: datetime


class SearchResult(BaseModel):
    document_id: str
    source_name: str
    path: str
    relative_path: str
    language: str
    size: int
    snippet: str
    rank: float


class ChunkSearchResult(BaseModel):
    chunk_id: str
    source_name: str
    relative_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    score: float


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _sanitize_fts_query(query: str) -> str:
    """Make query safe for FTS5 MATCH."""
    special = set('*(){}[]"^~:')
    if any(c in special for c in query):
        escaped = query.replace('"', '""')
        return f'"{escaped}"'
    return query.strip() or '""'


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------- #
#  Store
# --------------------------------------------------------------------------- #


class ContentStore:
    """Async SQLite store for indexed content with FTS5 search."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or Path.home() / ".helios"
        self._db_path = self._data_dir / "index.db"
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if not self._db:
            await self.initialize()
        assert self._db is not None
        return self._db

    # ----- Sources ----- #

    async def add_source(
        self, name: str, path: str, source_type: str = "directory",
    ) -> str:
        """Create or update a source. Returns source ID."""
        db = await self._ensure_db()
        now = datetime.now().isoformat()

        async with db.execute("SELECT id FROM sources WHERE name = ?", (name,)) as cur:
            row = await cur.fetchone()
            if row:
                source_id = row[0]
                await db.execute(
                    "UPDATE sources SET path = ?, updated_at = ? WHERE id = ?",
                    (path, now, source_id),
                )
                await db.commit()
                return source_id

        source_id = str(uuid4())
        await db.execute(
            """INSERT INTO sources (id, name, source_type, path, indexed_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source_id, name, source_type, path, now, now),
        )
        await db.commit()
        return source_id

    async def list_sources(self) -> list[Source]:
        db = await self._ensure_db()
        sources: list[Source] = []
        async with db.execute("SELECT * FROM sources ORDER BY name") as cur:
            async for row in cur:
                sources.append(Source(
                    id=row[0], name=row[1], source_type=row[2], path=row[3],
                    file_count=row[4], total_size=row[5],
                    indexed_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                ))
        return sources

    async def remove_source(self, name: str) -> bool:
        """Remove a source and all its documents."""
        db = await self._ensure_db()
        async with db.execute("SELECT id FROM sources WHERE name = ?", (name,)) as cur:
            row = await cur.fetchone()
            if not row:
                return False
            source_id = row[0]

        # Delete FTS entries
        async with db.execute(
            "SELECT id FROM documents WHERE source_id = ?", (source_id,),
        ) as cur:
            async for doc_row in cur:
                await db.execute(
                    "DELETE FROM documents_fts WHERE document_id = ?", (doc_row[0],),
                )

        # Delete chunk FTS entries
        async with db.execute(
            "SELECT id FROM chunks WHERE source_id = ?", (source_id,),
        ) as cur:
            async for chunk_row in cur:
                await db.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_row[0],),
                )
        await db.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))

        await db.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
        await db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        await db.commit()
        return True

    # ----- Documents ----- #

    async def upsert_document(
        self,
        source_id: str,
        path: str,
        relative_path: str,
        content: str,
        language: str = "",
        size: int = 0,
    ) -> bool:
        """Insert or update a document. Returns True if content changed."""
        db = await self._ensure_db()
        doc_hash = _content_hash(content)
        now = datetime.now().isoformat()

        async with db.execute(
            "SELECT id, content_hash FROM documents WHERE source_id = ? AND path = ?",
            (source_id, path),
        ) as cur:
            row = await cur.fetchone()

        if row:
            if row[1] == doc_hash:
                return False  # unchanged
            doc_id = row[0]
            await db.execute(
                """UPDATE documents SET content = ?, language = ?, size = ?,
                   content_hash = ?, indexed_at = ?, relative_path = ?
                   WHERE id = ?""",
                (content, language, size, doc_hash, now, relative_path, doc_id),
            )
            await db.execute(
                "DELETE FROM documents_fts WHERE document_id = ?", (doc_id,),
            )
            await db.execute(
                "INSERT INTO documents_fts (document_id, relative_path, content) VALUES (?, ?, ?)",
                (doc_id, relative_path, content),
            )
        else:
            doc_id = str(uuid4())
            await db.execute(
                """INSERT INTO documents
                   (id, source_id, path, relative_path, content, language, size, content_hash, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, source_id, path, relative_path, content, language, size, doc_hash, now),
            )
            await db.execute(
                "INSERT INTO documents_fts (document_id, relative_path, content) VALUES (?, ?, ?)",
                (doc_id, relative_path, content),
            )

        await db.commit()
        return True

    async def remove_stale_documents(
        self, source_id: str, current_paths: set[str],
    ) -> int:
        """Remove documents whose paths are no longer in the source."""
        db = await self._ensure_db()
        stale_ids: list[str] = []
        async with db.execute(
            "SELECT id, path FROM documents WHERE source_id = ?", (source_id,),
        ) as cur:
            async for row in cur:
                if row[1] not in current_paths:
                    stale_ids.append(row[0])

        for doc_id in stale_ids:
            await db.execute("DELETE FROM documents_fts WHERE document_id = ?", (doc_id,))
            await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        if stale_ids:
            await db.commit()
        return len(stale_ids)

    async def update_source_stats(self, source_id: str) -> None:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM documents WHERE source_id = ?",
            (source_id,),
        ) as cur:
            row = await cur.fetchone()
            assert row is not None
            file_count, total_size = row[0], row[1]

        await db.execute(
            "UPDATE sources SET file_count = ?, total_size = ?, updated_at = ? WHERE id = ?",
            (file_count, total_size, datetime.now().isoformat(), source_id),
        )
        await db.commit()

    async def get_document_by_path(self, path: str) -> Document | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM documents WHERE path = ?", (path,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return Document(
                id=row[0], source_id=row[1], path=row[2], relative_path=row[3],
                content=row[4], language=row[5], size=row[6],
                content_hash=row[7], indexed_at=datetime.fromisoformat(row[8]),
            )

    async def find_document_by_relative_path(
        self, relative_path: str, source_name: str | None = None,
    ) -> Document | None:
        """Find a document by relative path, optionally within a specific source."""
        db = await self._ensure_db()

        if source_name:
            async with db.execute(
                """SELECT d.* FROM documents d
                   JOIN sources s ON d.source_id = s.id
                   WHERE s.name = ? AND d.relative_path = ?""",
                (source_name, relative_path),
            ) as cur:
                row = await cur.fetchone()
        else:
            async with db.execute(
                "SELECT * FROM documents WHERE relative_path = ?", (relative_path,),
            ) as cur:
                row = await cur.fetchone()

        if not row:
            return None
        return Document(
            id=row[0], source_id=row[1], path=row[2], relative_path=row[3],
            content=row[4], language=row[5], size=row[6],
            content_hash=row[7], indexed_at=datetime.fromisoformat(row[8]),
        )

    async def get_source_files(
        self, source_name: str, pattern: str | None = None,
    ) -> list[str]:
        """List file paths in a source, optionally filtered by glob pattern."""
        db = await self._ensure_db()
        async with db.execute("SELECT id FROM sources WHERE name = ?", (source_name,)) as cur:
            row = await cur.fetchone()
            if not row:
                return []
            source_id = row[0]

        paths: list[str] = []
        async with db.execute(
            "SELECT relative_path FROM documents WHERE source_id = ? ORDER BY relative_path",
            (source_id,),
        ) as cur:
            async for row in cur:
                if pattern:
                    from fnmatch import fnmatch
                    if not fnmatch(row[0], pattern):
                        continue
                paths.append(row[0])
        return paths

    # ----- Chunks ----- #

    async def sync_document_chunks(
        self,
        document_id: str,
        source_id: str,
        chunks: list[tuple[str, int, int]],  # (content, start_line, end_line)
    ) -> tuple[int, int]:
        """Sync chunks for a document, preserving embeddings for unchanged content.

        Returns (created, removed) counts.
        """
        db = await self._ensure_db()

        # Get existing chunks keyed by content hash
        existing: dict[str, str] = {}  # content_hash -> chunk_id
        async with db.execute(
            "SELECT id, content_hash FROM chunks WHERE document_id = ?", (document_id,),
        ) as cur:
            async for row in cur:
                existing[row[1]] = row[0]

        # Determine what to create/remove
        new_hashes: set[str] = set()
        to_create: list[tuple[str, int, int, str]] = []  # (content, start, end, hash)
        for content, start, end in chunks:
            h = _content_hash(content)
            new_hashes.add(h)
            if h not in existing:
                to_create.append((content, start, end, h))

        # Remove stale chunks
        to_remove = set(existing.keys()) - new_hashes
        removed = 0
        for h in to_remove:
            chunk_id = existing[h]
            await db.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
            await db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            removed += 1

        # Create new chunks
        created = 0
        for content, start, end, h in to_create:
            chunk_id = str(uuid4())
            await db.execute(
                """INSERT INTO chunks
                   (id, document_id, source_id, content, content_hash, start_line, end_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, document_id, source_id, content, h, start, end),
            )
            await db.execute(
                "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
                (chunk_id, content),
            )
            created += 1

        if removed or created:
            await db.commit()
        return created, removed

    async def get_chunks_without_embeddings(
        self, source_id: str,
    ) -> list[tuple[str, str]]:
        """Get (chunk_id, content) for chunks that need embeddings."""
        db = await self._ensure_db()
        results: list[tuple[str, str]] = []
        async with db.execute(
            "SELECT id, content FROM chunks WHERE source_id = ? AND embedding IS NULL",
            (source_id,),
        ) as cur:
            async for row in cur:
                results.append((row[0], row[1]))
        return results

    async def store_embeddings(
        self, chunk_embeddings: list[tuple[str, list[float]]],
    ) -> None:
        """Store embeddings for chunks."""
        db = await self._ensure_db()
        for chunk_id, embedding in chunk_embeddings:
            blob = _serialize_f32(embedding)
            await db.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (blob, chunk_id),
            )
        await db.commit()

    async def get_chunk_count(self, source_id: str) -> int:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    async def get_embedding_count(self, source_id: str) -> int:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_id = ? AND embedding IS NOT NULL",
            (source_id,),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    async def get_documents_for_source(self, source_id: str) -> list[Document]:
        """Get all documents for a source."""
        db = await self._ensure_db()
        docs: list[Document] = []
        async with db.execute(
            "SELECT * FROM documents WHERE source_id = ? ORDER BY relative_path",
            (source_id,),
        ) as cur:
            async for row in cur:
                docs.append(Document(
                    id=row[0], source_id=row[1], path=row[2], relative_path=row[3],
                    content=row[4], language=row[5], size=row[6],
                    content_hash=row[7], indexed_at=datetime.fromisoformat(row[8]),
                ))
        return docs

    # ----- Chunk Search ----- #

    async def fts_chunk_search(
        self,
        query: str,
        source_name: str | None = None,
        language: str | None = None,
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        """Full-text search across chunks."""
        db = await self._ensure_db()
        safe_query = _sanitize_fts_query(query)
        overfetch = limit * 3 if (source_name or language) else limit

        results: list[ChunkSearchResult] = []
        async with db.execute(
            """SELECT chunk_id,
                      snippet(chunks_fts, 1, '', '', '...', 80) as snippet,
                      rank
               FROM chunks_fts
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, overfetch),
        ) as cur:
            async for row in cur:
                chunk_id, snippet, rank = row
                async with db.execute(
                    """SELECT c.content, c.start_line, c.end_line,
                              d.relative_path, d.language, s.name
                       FROM chunks c
                       JOIN documents d ON c.document_id = d.id
                       JOIN sources s ON c.source_id = s.id
                       WHERE c.id = ?""",
                    (chunk_id,),
                ) as detail_cur:
                    detail = await detail_cur.fetchone()
                    if not detail:
                        continue
                    if source_name and detail[5] != source_name:
                        continue
                    if language and detail[4] != language:
                        continue
                    results.append(ChunkSearchResult(
                        chunk_id=chunk_id,
                        source_name=detail[5],
                        relative_path=detail[3],
                        language=detail[4],
                        content=detail[0],
                        start_line=detail[1],
                        end_line=detail[2],
                        score=abs(rank),
                    ))
                if len(results) >= limit:
                    break

        return results

    async def vector_search(
        self,
        query_embedding: list[float],
        source_name: str | None = None,
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        """Search chunks by vector similarity (cosine)."""
        db = await self._ensure_db()

        if source_name:
            sql = """SELECT c.id, c.embedding
                     FROM chunks c
                     JOIN sources s ON c.source_id = s.id
                     WHERE s.name = ? AND c.embedding IS NOT NULL"""
            params: tuple = (source_name,)
        else:
            sql = "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
            params = ()

        # Compute similarities
        scored: list[tuple[float, str]] = []
        async with db.execute(sql, params) as cur:
            async for row in cur:
                emb = _deserialize_f32(row[1])
                sim = _cosine_similarity(query_embedding, emb)
                scored.append((sim, row[0]))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Fetch details for top results
        results: list[ChunkSearchResult] = []
        for sim, chunk_id in scored[:limit]:
            async with db.execute(
                """SELECT c.content, c.start_line, c.end_line,
                          d.relative_path, d.language, s.name
                   FROM chunks c
                   JOIN documents d ON c.document_id = d.id
                   JOIN sources s ON c.source_id = s.id
                   WHERE c.id = ?""",
                (chunk_id,),
            ) as cur:
                row = await cur.fetchone()
                if row:
                    results.append(ChunkSearchResult(
                        chunk_id=chunk_id,
                        source_name=row[5],
                        relative_path=row[3],
                        language=row[4],
                        content=row[0],
                        start_line=row[1],
                        end_line=row[2],
                        score=sim,
                    ))

        return results

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        source_name: str | None = None,
        language: str | None = None,
        limit: int = 10,
        rrf_k: int = 60,
    ) -> list[ChunkSearchResult]:
        """Hybrid search combining FTS5 keyword and vector semantic results.

        Uses Reciprocal Rank Fusion (RRF) to merge rankings.
        Falls back to FTS-only if no query embedding provided.
        """
        fts_results = await self.fts_chunk_search(
            query, source_name=source_name, language=language, limit=limit * 2,
        )

        vec_results: list[ChunkSearchResult] = []
        if query_embedding:
            vec_results = await self.vector_search(
                query_embedding, source_name=source_name, limit=limit * 2,
            )

        if not fts_results and not vec_results:
            return []
        if not vec_results:
            return fts_results[:limit]
        if not fts_results:
            return vec_results[:limit]

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, ChunkSearchResult] = {}

        for rank, r in enumerate(fts_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            chunk_data[r.chunk_id] = r

        for rank, r in enumerate(vec_results):
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            if r.chunk_id not in chunk_data:
                chunk_data[r.chunk_id] = r

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results: list[ChunkSearchResult] = []
        for cid in sorted_ids[:limit]:
            r = chunk_data[cid]
            results.append(ChunkSearchResult(
                chunk_id=r.chunk_id,
                source_name=r.source_name,
                relative_path=r.relative_path,
                language=r.language,
                content=r.content,
                start_line=r.start_line,
                end_line=r.end_line,
                score=rrf_scores[cid],
            ))

        return results

    async def has_chunks(self) -> bool:
        """Check if any chunks exist in the store."""
        db = await self._ensure_db()
        async with db.execute("SELECT COUNT(*) FROM chunks LIMIT 1") as cur:
            row = await cur.fetchone()
            return (row[0] if row else 0) > 0

    # ----- Document Search (legacy) ----- #

    async def search(
        self,
        query: str,
        source_name: str | None = None,
        language: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Full-text search across indexed documents."""
        db = await self._ensure_db()
        safe_query = _sanitize_fts_query(query)
        overfetch = limit * 3 if (source_name or language) else limit

        results: list[SearchResult] = []
        async with db.execute(
            """SELECT document_id,
                      snippet(documents_fts, 2, '>>>', '<<<', '...', 64) as snippet,
                      rank
               FROM documents_fts
               WHERE documents_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, overfetch),
        ) as cur:
            async for row in cur:
                doc_id, snippet, rank = row
                async with db.execute(
                    """SELECT d.source_id, d.path, d.relative_path, d.language, d.size, s.name
                       FROM documents d JOIN sources s ON d.source_id = s.id
                       WHERE d.id = ?""",
                    (doc_id,),
                ) as doc_cur:
                    doc_row = await doc_cur.fetchone()
                    if not doc_row:
                        continue
                    if source_name and doc_row[5] != source_name:
                        continue
                    if language and doc_row[3] != language:
                        continue
                    results.append(SearchResult(
                        document_id=doc_id,
                        source_name=doc_row[5],
                        path=doc_row[1],
                        relative_path=doc_row[2],
                        language=doc_row[3],
                        size=doc_row[4],
                        snippet=snippet,
                        rank=rank,
                    ))
                if len(results) >= limit:
                    break

        return results
