"""API surface extraction from indexed documents.

Regex-based extraction of API endpoints for common frameworks:
Next.js App Router, FastAPI, NestJS, Express.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ApiEndpoint:
    """A single API endpoint extracted from source code."""

    method: str  # GET, POST, PUT, DELETE, PATCH
    path: str  # /api/users/:id
    handler: str  # function/method name
    file_path: str  # relative path in project
    line_number: int  # 0-indexed
    request_type: str | None = None
    response_type: str | None = None
    framework: str = ""  # nextjs, fastapi, nestjs, express


@dataclass
class ApiSurface:
    """Extracted API surface for a project."""

    endpoints: list[ApiEndpoint] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Next.js App Router
# --------------------------------------------------------------------------- #

_NEXTJS_ROUTE_FILE = re.compile(
    r"(?:^|/)app/api/(.+)/route\.[tj]sx?$"
)

_NEXTJS_EXPORT_FN = re.compile(
    r"^export\s+(?:async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b",
    re.MULTILINE,
)

_NEXTJS_EXPORT_CONST = re.compile(
    r"^export\s+const\s+(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s*=",
    re.MULTILINE,
)


def _nextjs_path_from_file(relative_path: str) -> str:
    """Convert Next.js route file path to API path.

    app/api/users/[id]/route.ts -> /api/users/:id
    app/api/auth/[...nextauth]/route.ts -> /api/auth/:nextauth*
    """
    m = _NEXTJS_ROUTE_FILE.search(relative_path)
    if not m:
        return ""
    segments = m.group(1)
    # [...param] -> :param* (catch-all)
    path = re.sub(r"\[\.\.\.(\w+)\]", r":\1*", segments)
    # [param] -> :param (dynamic segment)
    path = re.sub(r"\[(\w+)\]", r":\1", path)
    return f"/api/{path}"


def _extract_nextjs(content: str, relative_path: str) -> list[ApiEndpoint]:
    """Extract endpoints from a Next.js App Router route file."""
    api_path = _nextjs_path_from_file(relative_path)
    if not api_path:
        return []

    endpoints: list[ApiEndpoint] = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        fn_match = _NEXTJS_EXPORT_FN.match(line.strip())
        const_match = _NEXTJS_EXPORT_CONST.match(line.strip()) if not fn_match else None
        match = fn_match or const_match
        if match:
            method = match.group(1)
            endpoints.append(ApiEndpoint(
                method=method,
                path=api_path,
                handler=method,
                file_path=relative_path,
                line_number=i,
                framework="nextjs",
            ))

    return endpoints


# --------------------------------------------------------------------------- #
#  FastAPI
# --------------------------------------------------------------------------- #

_FASTAPI_DECORATOR = re.compile(
    r"^@(?:app|router|api_router)\.(get|post|put|delete|patch|head|options)"
    r"\s*\(\s*[\"']([^\"']*)[\"']",
    re.MULTILINE | re.IGNORECASE,
)

_FASTAPI_HANDLER = re.compile(
    r"^(?:async\s+)?def\s+(\w+)\s*\(",
    re.MULTILINE,
)

_FASTAPI_RESPONSE_MODEL = re.compile(
    r"response_model\s*=\s*(\w+)",
)

_FASTAPI_BODY_PARAM = re.compile(
    r"(\w+)\s*:\s*([A-Z]\w+)",
)

_FASTAPI_ROUTER_PREFIX = re.compile(
    r"(?:APIRouter|FastAPI)\s*\(\s*[^)]*prefix\s*=\s*[\"']([^\"']+)[\"']",
)


def _extract_fastapi(content: str, relative_path: str) -> list[ApiEndpoint]:
    """Extract endpoints from a FastAPI file."""
    endpoints: list[ApiEndpoint] = []

    # Detect router prefix
    prefix_match = _FASTAPI_ROUTER_PREFIX.search(content)
    prefix = prefix_match.group(1) if prefix_match else ""

    for match in _FASTAPI_DECORATOR.finditer(content):
        method = match.group(1).upper()
        path = match.group(2)
        full_path = prefix + path

        # Find line number
        line_num = content[:match.start()].count("\n")

        # Find handler name (next def after decorator)
        after = content[match.end():]
        handler_match = _FASTAPI_HANDLER.search(after)
        handler = handler_match.group(1) if handler_match else "unknown"

        # Extract response model from decorator
        decorator_text = content[match.start():match.end() + 200]
        response_match = _FASTAPI_RESPONSE_MODEL.search(decorator_text)
        response_type = response_match.group(1) if response_match else None

        # Extract request body type from handler signature
        request_type = None
        if handler_match:
            sig_start = match.end() + handler_match.start()
            sig_end = content.find(")", sig_start) + 1
            if sig_end > sig_start:
                sig = content[sig_start:sig_end]
                for bp in _FASTAPI_BODY_PARAM.finditer(sig):
                    _, type_name = bp.group(1), bp.group(2)
                    # Skip common non-body types
                    if type_name not in ("Request", "Response", "Depends", "Query", "Path", "Header"):
                        request_type = type_name
                        break

        endpoints.append(ApiEndpoint(
            method=method,
            path=full_path,
            handler=handler,
            file_path=relative_path,
            line_number=line_num,
            request_type=request_type,
            response_type=response_type,
            framework="fastapi",
        ))

    return endpoints


# --------------------------------------------------------------------------- #
#  NestJS
# --------------------------------------------------------------------------- #

_NESTJS_CONTROLLER = re.compile(
    r"@Controller\s*\(\s*['\"]([^'\"]*)['\"]",
)

_NESTJS_METHOD = re.compile(
    r"@(Get|Post|Put|Delete|Patch|Head|Options)\s*\(\s*(?:['\"]([^'\"]*)['\"])?\s*\)",
)

_NESTJS_HANDLER = re.compile(
    r"(?:async\s+)?(\w+)\s*\(",
)

_NESTJS_BODY_DTO = re.compile(
    r"@Body\(\)\s+\w+\s*:\s*(\w+)",
)


def _extract_nestjs(content: str, relative_path: str) -> list[ApiEndpoint]:
    """Extract endpoints from a NestJS controller file."""
    endpoints: list[ApiEndpoint] = []

    # Find controller base path
    ctrl_match = _NESTJS_CONTROLLER.search(content)
    base_path = "/" + ctrl_match.group(1).strip("/") if ctrl_match else ""

    for match in _NESTJS_METHOD.finditer(content):
        method = match.group(1).upper()
        sub_path = match.group(2) or ""
        if sub_path and not sub_path.startswith("/"):
            sub_path = "/" + sub_path
        full_path = base_path + sub_path

        line_num = content[:match.start()].count("\n")

        # Find handler name (next method-like after decorator)
        after = content[match.end():]
        handler_match = _NESTJS_HANDLER.search(after)
        handler = handler_match.group(1) if handler_match else "unknown"

        # Find @Body() DTO near this method
        request_type = None
        search_region = content[match.end():match.end() + 300]
        dto_match = _NESTJS_BODY_DTO.search(search_region)
        if dto_match:
            request_type = dto_match.group(1)

        endpoints.append(ApiEndpoint(
            method=method,
            path=full_path,
            handler=handler,
            file_path=relative_path,
            line_number=line_num,
            request_type=request_type,
            framework="nestjs",
        ))

    return endpoints


# --------------------------------------------------------------------------- #
#  Express
# --------------------------------------------------------------------------- #

_EXPRESS_ROUTE = re.compile(
    r"(?:app|router)\.(get|post|put|delete|patch|all|head|options)"
    r"\s*\(\s*['\"]([^'\"]+)['\"]"
    r"(?:\s*,\s*(\w+))?",
    re.IGNORECASE,
)


def _extract_express(content: str, relative_path: str) -> list[ApiEndpoint]:
    """Extract endpoints from an Express route file."""
    endpoints: list[ApiEndpoint] = []

    for match in _EXPRESS_ROUTE.finditer(content):
        method = match.group(1).upper()
        path = match.group(2)
        handler = match.group(3) or "anonymous"
        line_num = content[:match.start()].count("\n")

        endpoints.append(ApiEndpoint(
            method=method,
            path=path,
            handler=handler,
            file_path=relative_path,
            line_number=line_num,
            framework="express",
        ))

    return endpoints


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


async def extract_api_surface(
    source_id: str,
    store,
) -> ApiSurface:
    """Extract API endpoints from all documents in an indexed source.

    Dispatches to framework-specific extractors based on file extension
    and content heuristics.
    """
    from helios.indexing.store import ContentStore

    assert isinstance(store, ContentStore)
    docs = await store.get_documents_for_source(source_id)

    surface = ApiSurface()
    frameworks_seen: set[str] = set()

    for doc in docs:
        rp = doc.relative_path
        content = doc.content
        extracted: list[ApiEndpoint] = []

        # Next.js App Router: route files
        if _NEXTJS_ROUTE_FILE.search(rp):
            extracted.extend(_extract_nextjs(content, rp))

        # FastAPI: Python files with decorator patterns
        if rp.endswith(".py") and re.search(
            r"@(?:app|router|api_router)\.(get|post|put|delete|patch)", content, re.IGNORECASE,
        ):
            extracted.extend(_extract_fastapi(content, rp))

        # NestJS: TypeScript files with @Controller
        if rp.endswith(".ts") and "@Controller" in content:
            extracted.extend(_extract_nestjs(content, rp))

        # Express: JS/TS files with app.get/router.get patterns
        if rp.endswith((".js", ".ts")) and re.search(
            r"(?:app|router)\.(get|post|put|delete|patch)\s*\(", content, re.IGNORECASE,
        ):
            # Avoid double-extraction if already handled by NestJS
            if not any(e.framework == "nestjs" for e in extracted):
                extracted.extend(_extract_express(content, rp))

        for ep in extracted:
            frameworks_seen.add(ep.framework)
        surface.endpoints.extend(extracted)

    surface.frameworks = sorted(frameworks_seen)
    return surface
