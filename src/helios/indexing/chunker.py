"""Document chunking for embedding and semantic search."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns that indicate a new top-level code definition
_CODE_BOUNDARY = re.compile(
    r"^(?:"
    r"(?:async\s+)?def\s+\w+"                                     # Python
    r"|class\s+\w+"                                                # Python/JS/TS/Java
    r"|(?:export\s+)?(?:function|const|let|var)\s+\w+"             # JS/TS
    r"|(?:export\s+)?(?:interface|type|enum)\s+\w+"                # TS
    r"|(?:pub\s+)?(?:fn|struct|enum|impl|trait|mod)\s+"            # Rust
    r"|func\s+"                                                    # Go
    r"|(?:public|private|protected|internal)\s+.*(?:class|void|int|string|async)" # Java/C#
    r")"
)

_HEADING = re.compile(r"^#{1,6}\s+")

CODE_LANGUAGES = {
    "python", "javascript", "typescript", "go", "rust", "java",
    "c", "cpp", "csharp", "ruby", "php", "swift", "kotlin",
    "scala", "shell", "elixir", "haskell", "dart", "zig", "fsharp",
    "jsx", "tsx",
}

MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 50


@dataclass
class Chunk:
    content: str
    start_line: int
    end_line: int


def chunk_document(
    content: str,
    language: str = "",
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split a document into chunks for embedding.

    Uses language-aware boundaries for code, heading-based splitting
    for markdown, and paragraph-based splitting for everything else.
    """
    if not content.strip():
        return []

    lines = content.split("\n")

    if language in CODE_LANGUAGES:
        raw = _split_at_boundaries(lines, _CODE_BOUNDARY, max_chars)
    elif language in ("markdown", "restructuredtext", "asciidoc"):
        raw = _split_at_boundaries(lines, _HEADING, max_chars)
    else:
        raw = _split_by_paragraphs(lines, max_chars)

    return _merge_small_chunks(raw, min_chars, max_chars)


def _split_at_boundaries(
    lines: list[str], pattern: re.Pattern[str], max_chars: int,
) -> list[Chunk]:
    """Split at lines matching a regex pattern (function/class defs, headings)."""
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = 0

    for i, line in enumerate(lines):
        if pattern.match(line) and current_lines:
            content = "\n".join(current_lines)
            chunks.append(Chunk(content=content, start_line=current_start, end_line=i - 1))
            current_lines = [line]
            current_start = i
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines)
        chunks.append(Chunk(content=content, start_line=current_start, end_line=len(lines) - 1))

    # Split oversized chunks
    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.content) > max_chars:
            result.extend(_split_oversized(chunk, max_chars))
        else:
            result.append(chunk)

    return result


def _split_by_paragraphs(lines: list[str], max_chars: int) -> list[Chunk]:
    """Split on blank lines (paragraph boundaries)."""
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = 0

    for i, line in enumerate(lines):
        if not line.strip() and current_lines:
            content = "\n".join(current_lines)
            if len(content) > max_chars:
                chunks.extend(_split_oversized(
                    Chunk(content=content, start_line=current_start, end_line=i - 1),
                    max_chars,
                ))
            else:
                chunks.append(Chunk(content=content, start_line=current_start, end_line=i - 1))
            current_lines = []
            current_start = i + 1
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines)
        chunks.append(Chunk(content=content, start_line=current_start, end_line=len(lines) - 1))

    return chunks


def _split_oversized(chunk: Chunk, max_chars: int) -> list[Chunk]:
    """Split an oversized chunk at line boundaries."""
    lines = chunk.content.split("\n")
    result: list[Chunk] = []
    current_lines: list[str] = []
    current_size = 0
    current_start = chunk.start_line

    for i, line in enumerate(lines):
        line_size = len(line) + 1
        if current_size + line_size > max_chars and current_lines:
            content = "\n".join(current_lines)
            result.append(Chunk(
                content=content,
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
            ))
            current_lines = [line]
            current_size = line_size
            current_start = chunk.start_line + i
        else:
            current_lines.append(line)
            current_size += line_size

    if current_lines:
        content = "\n".join(current_lines)
        result.append(Chunk(
            content=content,
            start_line=current_start,
            end_line=chunk.end_line,
        ))

    return result


def _merge_small_chunks(
    chunks: list[Chunk], min_chars: int, max_chars: int,
) -> list[Chunk]:
    """Merge consecutive small chunks to avoid tiny embeddings."""
    if not chunks:
        return []

    result: list[Chunk] = []
    current = chunks[0]

    for next_chunk in chunks[1:]:
        merged_content = current.content + "\n\n" + next_chunk.content
        if len(current.content) < min_chars and len(merged_content) <= max_chars:
            current = Chunk(
                content=merged_content,
                start_line=current.start_line,
                end_line=next_chunk.end_line,
            )
        else:
            if len(current.content) >= min_chars:
                result.append(current)
            else:
                # Tiny trailing chunk — merge with previous if possible
                if result:
                    prev = result[-1]
                    merged = prev.content + "\n\n" + current.content
                    if len(merged) <= max_chars:
                        result[-1] = Chunk(
                            content=merged,
                            start_line=prev.start_line,
                            end_line=current.end_line,
                        )
                    else:
                        result.append(current)
                else:
                    result.append(current)
            current = next_chunk

    # Always keep the last chunk
    result.append(current)
    return result
