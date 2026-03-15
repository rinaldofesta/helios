"""Project stack analysis — FE libraries, types/schemas, API clients, conventions.

Analyzes indexed project documents to detect the full technology stack,
extract type definitions, find existing API clients, and detect project patterns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data models
# --------------------------------------------------------------------------- #


@dataclass
class Library:
    """A detected project library/dependency."""

    name: str
    version: str
    category: str  # ui_framework, css, state, data_fetching, etc.


@dataclass
class TypeDef:
    """An extracted type/schema definition."""

    name: str
    file_path: str  # relative path
    line_number: int
    kind: str  # interface, type, zod_schema, pydantic_model
    preview: str  # first ~200 chars of the definition


@dataclass
class ApiClient:
    """A detected API client/wrapper."""

    name: str
    file_path: str
    kind: str  # axios, fetch_wrapper, trpc, react_query_hooks, api_service
    description: str


@dataclass
class ProjectPatterns:
    """Detected project conventions and patterns."""

    folder_structure: dict[str, str]  # folder -> purpose
    component_style: str  # e.g. "function components with named exports"
    naming_convention: str  # e.g. "kebab-case files, PascalCase components"
    key_directories: list[str]  # important directories found


@dataclass
class StackAnalysis:
    """Complete project stack analysis result."""

    libraries: list[Library] = field(default_factory=list)
    type_defs: list[TypeDef] = field(default_factory=list)
    api_clients: list[ApiClient] = field(default_factory=list)
    patterns: ProjectPatterns | None = None


# --------------------------------------------------------------------------- #
#  FE Library Detection
# --------------------------------------------------------------------------- #

# Maps package names to categories for the CLAUDE.md
LIBRARY_CATEGORIES: dict[str, list[str]] = {
    "ui_framework": [
        "react", "react-dom", "vue", "svelte", "@angular/core", "solid-js", "preact",
    ],
    "meta_framework": [
        "next", "nuxt", "@sveltejs/kit", "@angular/cli", "gatsby", "remix",
        "@remix-run/react", "astro",
    ],
    "ui_library": [
        "@radix-ui/react-slot", "@radix-ui/themes",
        "@mui/material", "@mui/joy",
        "@chakra-ui/react",
        "antd",
        "@mantine/core",
        "@headlessui/react",
        "class-variance-authority",  # shadcn indicator
    ],
    "css": [
        "tailwindcss", "styled-components", "@emotion/react", "@emotion/styled",
        "sass", "less", "postcss", "autoprefixer", "clsx", "tailwind-merge",
    ],
    "state_management": [
        "zustand", "redux", "@reduxjs/toolkit", "jotai", "recoil", "mobx",
        "valtio", "xstate", "@xstate/react", "nanostores",
    ],
    "data_fetching": [
        "@tanstack/react-query", "swr", "@trpc/client", "@trpc/react-query",
        "@apollo/client", "graphql-request", "urql",
    ],
    "forms": [
        "react-hook-form", "@hookform/resolvers", "formik",
    ],
    "validation": [
        "zod", "yup", "joi", "superstruct", "valibot",
    ],
    "auth": [
        "next-auth", "@auth/core", "@clerk/nextjs", "@clerk/clerk-react",
        "@supabase/supabase-js", "@supabase/auth-helpers-nextjs",
        "firebase", "@firebase/auth",
    ],
    "animation": [
        "framer-motion", "motion", "gsap", "@react-spring/web",
        "react-transition-group",
    ],
    "icons": [
        "lucide-react", "react-icons", "@heroicons/react",
        "@phosphor-icons/react", "@tabler/icons-react",
    ],
    "dates": [
        "date-fns", "dayjs", "luxon", "moment",
    ],
    "tables": [
        "@tanstack/react-table", "react-data-grid", "ag-grid-react",
    ],
    "toast_notifications": [
        "sonner", "react-hot-toast", "react-toastify", "notistack",
    ],
    "testing": [
        "vitest", "jest", "@testing-library/react", "cypress", "playwright",
    ],
}

# Reverse lookup: package name -> category
_PACKAGE_TO_CATEGORY: dict[str, str] = {}
for _cat, _pkgs in LIBRARY_CATEGORIES.items():
    for _pkg in _pkgs:
        _PACKAGE_TO_CATEGORY[_pkg] = _cat

# shadcn detection: if these packages are ALL present, it's likely shadcn
_SHADCN_INDICATORS = {"tailwindcss", "class-variance-authority", "@radix-ui/react-slot"}


def detect_fe_libraries(project_path: Path) -> list[Library]:
    """Detect FE libraries from package.json files."""
    libraries: list[Library] = []
    seen: set[str] = set()

    # Search for package.json in root and common locations
    candidates = [
        project_path / "package.json",
        project_path / "frontend" / "package.json",
        project_path / "client" / "package.json",
        project_path / "apps" / "web" / "package.json",
        project_path / "packages" / "ui" / "package.json",
    ]

    for pkg_path in candidates:
        if not pkg_path.exists():
            continue
        try:
            data = json.loads(pkg_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        all_deps: dict[str, str] = {}
        all_deps.update(data.get("dependencies", {}))
        all_deps.update(data.get("devDependencies", {}))

        for name, version in all_deps.items():
            if name in seen:
                continue
            category = _PACKAGE_TO_CATEGORY.get(name)
            # Also check prefix matches for scoped packages like @radix-ui/*
            if not category:
                for prefix in ("@radix-ui/", "@mui/", "@chakra-ui/", "@mantine/"):
                    if name.startswith(prefix):
                        category = "ui_library"
                        break
            if category:
                libraries.append(Library(
                    name=name,
                    version=version.lstrip("^~>=<"),
                    category=category,
                ))
                seen.add(name)

    # Detect shadcn (it's not a package, it's a pattern)
    lib_names = {lib.name for lib in libraries}
    if _SHADCN_INDICATORS.issubset(lib_names):
        # Check for components.json (shadcn config)
        if (project_path / "components.json").exists():
            libraries.append(Library(name="shadcn/ui", version="", category="ui_library"))

    return sorted(libraries, key=lambda lib: (lib.category, lib.name))


# --------------------------------------------------------------------------- #
#  Type/Schema Extraction
# --------------------------------------------------------------------------- #

_TS_INTERFACE = re.compile(
    r"^export\s+interface\s+(\w+)\s*(?:extends\s+\w+(?:\s*,\s*\w+)*)?\s*\{",
    re.MULTILINE,
)

_TS_TYPE = re.compile(
    r"^export\s+type\s+(\w+)\s*=",
    re.MULTILINE,
)

_ZOD_SCHEMA = re.compile(
    r"^export\s+const\s+(\w+)\s*=\s*z\.",
    re.MULTILINE,
)

_PYDANTIC_MODEL = re.compile(
    r"^class\s+(\w+)\s*\(\s*(?:BaseModel|BaseSchema|BaseSettings)",
    re.MULTILINE,
)

# Heuristic: these types are likely API-related
_API_TYPE_HINTS = re.compile(
    r"(?:Request|Response|Dto|Schema|Payload|Input|Output|Params|Body)",
    re.IGNORECASE,
)


def _extract_definition_preview(content: str, match_start: int, max_chars: int = 200) -> str:
    """Extract a preview of a type definition starting from match position."""
    # Find the end of the definition (matching braces or next export)
    text = content[match_start:]
    brace_count = 0
    end = 0
    for i, ch in enumerate(text):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
        if i > max_chars * 2:  # safety limit
            end = max_chars
            break
    if end == 0:
        end = min(len(text), max_chars)
    preview = text[:end].strip()
    if len(preview) > max_chars:
        preview = preview[:max_chars] + "..."
    return preview


async def extract_type_defs(source_id: str, store) -> list[TypeDef]:
    """Extract type/schema definitions from indexed documents."""
    docs = await store.get_documents_for_source(source_id)
    type_defs: list[TypeDef] = []

    for doc in docs:
        rp = doc.relative_path
        content = doc.content

        # TypeScript interfaces
        if rp.endswith((".ts", ".tsx")):
            for match in _TS_INTERFACE.finditer(content):
                name = match.group(1)
                line = content[:match.start()].count("\n")
                preview = _extract_definition_preview(content, match.start())
                type_defs.append(TypeDef(
                    name=name, file_path=rp, line_number=line,
                    kind="interface", preview=preview,
                ))

            for match in _TS_TYPE.finditer(content):
                name = match.group(1)
                line = content[:match.start()].count("\n")
                # Get the full type (until semicolon or next export)
                text_after = content[match.start():]
                end = text_after.find(";")
                if end == -1:
                    end = min(200, len(text_after))
                preview = text_after[:end + 1].strip()
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                type_defs.append(TypeDef(
                    name=name, file_path=rp, line_number=line,
                    kind="type", preview=preview,
                ))

        # Zod schemas (TS files)
        if rp.endswith((".ts", ".tsx")):
            for match in _ZOD_SCHEMA.finditer(content):
                name = match.group(1)
                line = content[:match.start()].count("\n")
                preview = _extract_definition_preview(content, match.start())
                type_defs.append(TypeDef(
                    name=name, file_path=rp, line_number=line,
                    kind="zod_schema", preview=preview,
                ))

        # Pydantic models (Python files)
        if rp.endswith(".py"):
            for match in _PYDANTIC_MODEL.finditer(content):
                name = match.group(1)
                line = content[:match.start()].count("\n")
                preview = _extract_definition_preview(content, match.start())
                type_defs.append(TypeDef(
                    name=name, file_path=rp, line_number=line,
                    kind="pydantic_model", preview=preview,
                ))

    # Sort: API-related types first, then alphabetical
    def sort_key(td: TypeDef) -> tuple[int, str]:
        is_api = 0 if _API_TYPE_HINTS.search(td.name) else 1
        return (is_api, td.name)

    return sorted(type_defs, key=sort_key)


# --------------------------------------------------------------------------- #
#  API Client Detection
# --------------------------------------------------------------------------- #

_AXIOS_CREATE = re.compile(r"axios\.create\s*\(")
_FETCH_WRAPPER = re.compile(
    r"export\s+(?:const|function|async\s+function)\s+(\w*(?:api|fetch|client|http)\w*)",
    re.IGNORECASE,
)
_TRPC_CLIENT = re.compile(r"createTRPC(?:Client|React|ProxyClient)|initTRPC")
_REACT_QUERY_HOOK = re.compile(r"export\s+(?:const|function)\s+(use\w+(?:Query|Mutation))")
_KY_CLIENT = re.compile(r"ky\.create\s*\(")


async def detect_api_clients(source_id: str, store) -> list[ApiClient]:
    """Detect existing API clients and wrappers in the project."""
    docs = await store.get_documents_for_source(source_id)
    clients: list[ApiClient] = []
    seen_files: set[str] = set()

    for doc in docs:
        rp = doc.relative_path
        content = doc.content

        if not rp.endswith((".ts", ".tsx", ".js", ".jsx")):
            continue

        # Axios instances
        if _AXIOS_CREATE.search(content):
            if rp not in seen_files:
                clients.append(ApiClient(
                    name="axios instance",
                    file_path=rp,
                    kind="axios",
                    description="Configured axios client — use this for API calls",
                ))
                seen_files.add(rp)

        # tRPC client
        if _TRPC_CLIENT.search(content):
            if rp not in seen_files:
                clients.append(ApiClient(
                    name="tRPC client",
                    file_path=rp,
                    kind="trpc",
                    description="tRPC client — use typed procedures instead of raw fetch",
                ))
                seen_files.add(rp)

        # Custom fetch wrappers
        for match in _FETCH_WRAPPER.finditer(content):
            fn_name = match.group(1)
            if rp not in seen_files:
                clients.append(ApiClient(
                    name=fn_name,
                    file_path=rp,
                    kind="fetch_wrapper",
                    description=f"Custom API wrapper `{fn_name}` — use instead of raw fetch()",
                ))
                seen_files.add(rp)

        # React Query hooks
        hooks_found: list[str] = []
        for match in _REACT_QUERY_HOOK.finditer(content):
            hooks_found.append(match.group(1))
        if hooks_found and rp not in seen_files:
            hooks_preview = ", ".join(hooks_found[:5])
            if len(hooks_found) > 5:
                hooks_preview += f" (+{len(hooks_found) - 5} more)"
            clients.append(ApiClient(
                name="React Query hooks",
                file_path=rp,
                kind="react_query_hooks",
                description=f"Data fetching hooks: {hooks_preview}",
            ))
            seen_files.add(rp)

        # Common API service files by name pattern
        name_lower = Path(rp).stem.lower()
        if name_lower in ("api", "client", "http", "service", "fetcher") and rp not in seen_files:
            clients.append(ApiClient(
                name=name_lower,
                file_path=rp,
                kind="api_service",
                description="API service file — check for existing methods before creating new ones",
            ))
            seen_files.add(rp)

    return clients


# --------------------------------------------------------------------------- #
#  Project Pattern Detection
# --------------------------------------------------------------------------- #

# Common FE project directories and their purposes
_KNOWN_DIRECTORIES: dict[str, str] = {
    "components": "Shared/reusable UI components",
    "ui": "Base UI primitives (often shadcn)",
    "features": "Feature-based modules",
    "pages": "Page components (Next.js pages router)",
    "app": "App router (Next.js 13+)",
    "hooks": "Custom React hooks",
    "lib": "Utility libraries and configurations",
    "utils": "Utility functions",
    "services": "API service layer",
    "api": "API routes or client code",
    "store": "State management",
    "stores": "State management stores",
    "types": "TypeScript type definitions",
    "interfaces": "TypeScript interfaces",
    "schemas": "Validation schemas (zod, yup)",
    "styles": "Global styles and themes",
    "constants": "App constants and config",
    "config": "Configuration files",
    "middleware": "Middleware functions",
    "providers": "React context providers",
    "contexts": "React contexts",
    "layouts": "Layout components",
    "templates": "Page templates",
    "assets": "Static assets (images, fonts)",
    "public": "Public static files",
    "__tests__": "Test files",
    "tests": "Test files",
    "e2e": "End-to-end tests",
}


async def detect_project_patterns(source_id: str, store) -> ProjectPatterns:
    """Detect project conventions from the indexed file structure."""
    docs = await store.get_documents_for_source(source_id)

    # Collect all relative paths
    all_paths = [doc.relative_path for doc in docs]

    # Detect folder structure
    folders_seen: set[str] = set()
    for rp in all_paths:
        parts = Path(rp).parts
        for i, part in enumerate(parts[:-1]):  # skip filename
            folder = part.lower()
            if folder in _KNOWN_DIRECTORIES and folder not in folders_seen:
                folders_seen.add(folder)

    folder_structure = {
        f: _KNOWN_DIRECTORIES[f] for f in sorted(folders_seen)
    }

    key_directories = sorted(folders_seen)

    # Detect naming convention from component files
    component_files = [
        Path(rp).stem for rp in all_paths
        if rp.endswith((".tsx", ".jsx"))
    ]

    pascal_count = sum(1 for f in component_files if f[0:1].isupper() and "_" not in f and "-" not in f)
    kebab_count = sum(1 for f in component_files if "-" in f)
    snake_count = sum(1 for f in component_files if "_" in f and "-" not in f)
    camel_count = sum(1 for f in component_files if f[0:1].islower() and f != f.lower() and "_" not in f and "-" not in f)

    total = pascal_count + kebab_count + snake_count + camel_count
    if total == 0:
        naming_convention = "not detected"
    else:
        counts = {
            "PascalCase": pascal_count,
            "kebab-case": kebab_count,
            "snake_case": snake_count,
            "camelCase": camel_count,
        }
        naming_convention = max(counts, key=counts.get)  # type: ignore[arg-type]

    # Detect component style
    component_style = "not detected"
    tsx_files = [doc for doc in docs if doc.relative_path.endswith((".tsx", ".jsx"))]
    if tsx_files:
        default_exports = sum(
            1 for doc in tsx_files[:20]
            if "export default" in doc.content
        )
        named_exports = sum(
            1 for doc in tsx_files[:20]
            if re.search(r"^export\s+(?:const|function)\s+\w+", doc.content, re.MULTILINE)
        )
        sample_size = min(len(tsx_files), 20)
        if default_exports > named_exports:
            component_style = f"default exports ({default_exports}/{sample_size} sampled files)"
        elif named_exports > 0:
            component_style = f"named exports ({named_exports}/{sample_size} sampled files)"

    return ProjectPatterns(
        folder_structure=folder_structure,
        component_style=component_style,
        naming_convention=naming_convention,
        key_directories=key_directories,
    )


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


async def analyze_stack(
    source_id: str,
    project_path: Path,
    store,
) -> StackAnalysis:
    """Run full stack analysis on an indexed project."""
    libraries = detect_fe_libraries(project_path)
    type_defs = await extract_type_defs(source_id, store)
    api_clients = await detect_api_clients(source_id, store)
    patterns = await detect_project_patterns(source_id, store)

    return StackAnalysis(
        libraries=libraries,
        type_defs=type_defs,
        api_clients=api_clients,
        patterns=patterns,
    )
