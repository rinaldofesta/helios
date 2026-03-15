"""Dependency detection and source path resolution.

Parses project manifest files (pyproject.toml, requirements.txt, package.json)
to discover dependencies, then finds their installed source code on disk.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_PACKAGE_FILES = 500


@dataclass
class Dependency:
    name: str
    version: str | None = None
    source_path: Path | None = None
    docs_url: str | None = None
    ecosystem: str = ""  # "python", "javascript"
    dev: bool = False


@dataclass
class DepsResult:
    """Result of dependency detection for a project."""
    project_path: Path
    ecosystems: list[str] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    resolved: int = 0
    unresolved: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def detect_dependencies(
    project_path: Path,
    include_dev: bool = False,
) -> DepsResult:
    """Detect project dependencies from manifest files."""
    result = DepsResult(project_path=project_path)
    deps: list[Dependency] = []

    # Python
    pyproject = project_path / "pyproject.toml"
    requirements = project_path / "requirements.txt"
    if pyproject.exists():
        deps.extend(_parse_pyproject(pyproject))
        result.ecosystems.append("python")
    elif requirements.exists():
        deps.extend(_parse_requirements(requirements))
        result.ecosystems.append("python")

    # JavaScript / TypeScript
    package_json = project_path / "package.json"
    if package_json.exists():
        deps.extend(_parse_package_json(package_json))
        result.ecosystems.append("javascript")

    # Filter dev deps
    if not include_dev:
        deps = [d for d in deps if not d.dev]

    # Resolve source paths
    py_site = _find_python_site_packages(project_path)
    node_modules = project_path / "node_modules"

    for dep in deps:
        if dep.ecosystem == "python" and py_site:
            dep.source_path = _find_python_pkg_dir(dep.name, py_site)
        elif dep.ecosystem == "javascript" and node_modules.is_dir():
            pkg_dir = node_modules / dep.name
            if pkg_dir.is_dir():
                dep.source_path = pkg_dir

        if dep.source_path:
            result.resolved += 1
        else:
            result.unresolved.append(dep.name)

    # Discover documentation URLs from package metadata
    discover_docs_urls(deps, py_site)

    result.dependencies = deps
    return result


# --------------------------------------------------------------------------- #
#  Python parsers
# --------------------------------------------------------------------------- #


def _parse_pyproject(path: Path) -> list[Dependency]:
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)

    deps: list[Dependency] = []

    # PEP 621 dependencies
    for req in data.get("project", {}).get("dependencies", []):
        name = _parse_requirement_name(req)
        if name:
            deps.append(Dependency(name=name, ecosystem="python"))

    # Optional/dev dependencies
    for group, group_deps in data.get("project", {}).get("optional-dependencies", {}).items():
        is_dev = group.lower() in ("dev", "test", "testing", "lint", "docs", "typing")
        for req in group_deps:
            name = _parse_requirement_name(req)
            if name:
                deps.append(Dependency(name=name, ecosystem="python", dev=is_dev))

    return deps


def _parse_requirements(path: Path) -> list[Dependency]:
    deps: list[Dependency] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = _parse_requirement_name(line)
        if name:
            deps.append(Dependency(name=name, ecosystem="python"))
    return deps


def _parse_requirement_name(req_str: str) -> str | None:
    """Extract package name from a PEP 508 requirement string."""
    match = re.match(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)", req_str.strip())
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
#  JavaScript parsers
# --------------------------------------------------------------------------- #


def _parse_package_json(path: Path) -> list[Dependency]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    deps: list[Dependency] = []
    for name, version in data.get("dependencies", {}).items():
        deps.append(Dependency(name=name, version=version, ecosystem="javascript"))
    for name, version in data.get("devDependencies", {}).items():
        deps.append(Dependency(name=name, version=version, ecosystem="javascript", dev=True))
    return deps


# --------------------------------------------------------------------------- #
#  Source path resolution
# --------------------------------------------------------------------------- #


def _find_python_site_packages(project_path: Path) -> Path | None:
    """Find site-packages in the project's virtual environment."""
    for venv_name in (".venv", "venv", "env", ".env"):
        venv_path = project_path / venv_name
        if not venv_path.is_dir():
            continue
        lib_path = venv_path / "lib"
        if not lib_path.is_dir():
            continue
        for python_dir in sorted(lib_path.iterdir(), reverse=True):
            if python_dir.name.startswith("python"):
                site_packages = python_dir / "site-packages"
                if site_packages.is_dir():
                    return site_packages
    return None


def _find_python_pkg_dir(name: str, site_packages: Path) -> Path | None:
    """Find a Python package directory in site-packages."""
    # Normalize: "pydantic-settings" -> "pydantic_settings"
    normalized = name.replace("-", "_").lower()

    # Direct match
    for entry in site_packages.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.endswith((".dist-info", ".egg-info", "__pycache__")):
            continue
        if entry.name.lower() == normalized:
            return entry

    # Fallback: check dist-info for top_level.txt
    for entry in site_packages.iterdir():
        if not entry.name.endswith(".dist-info") or not entry.is_dir():
            continue
        dist_name = entry.name.split("-")[0].lower().replace("-", "_")
        if dist_name != normalized:
            continue
        top_level = entry / "top_level.txt"
        if top_level.exists():
            pkg_name = top_level.read_text().strip().split("\n")[0].strip()
            if pkg_name:
                pkg_dir = site_packages / pkg_name
                if pkg_dir.is_dir():
                    return pkg_dir

    return None


def discover_docs_urls(deps: list[Dependency], site_packages: Path | None) -> None:
    """Try to find documentation URLs for Python packages from dist-info metadata."""
    if not site_packages:
        return

    for dep in deps:
        if dep.ecosystem != "python" or dep.docs_url:
            continue

        normalized = dep.name.replace("-", "_").lower()

        for entry in site_packages.iterdir():
            if not entry.name.endswith(".dist-info") or not entry.is_dir():
                continue
            dist_name = entry.name.split("-")[0].lower().replace("-", "_")
            if dist_name != normalized:
                continue

            # Read METADATA file for Project-URL entries
            metadata_file = entry / "METADATA"
            if not metadata_file.exists():
                break

            try:
                text = metadata_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                break

            for line in text.splitlines():
                if not line.startswith("Project-URL:"):
                    continue
                # Format: "Project-URL: Documentation, https://..."
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                label = parts[0].replace("Project-URL:", "").strip().lower()
                url = parts[1].strip()
                if label in ("documentation", "docs", "doc", "homepage", "home"):
                    if url.startswith("http"):
                        dep.docs_url = url
                        # Prefer "documentation" over "homepage"
                        if label in ("documentation", "docs", "doc"):
                            break
            break


def estimate_package_size(path: Path) -> int:
    """Count the number of indexable files in a package directory."""
    from helios.indexing.scanner import detect_language, IGNORE_DIRS

    count = 0
    for dirpath, dirnames, filenames in __import__("os").walk(path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in filenames:
            if detect_language(Path(f)):
                count += 1
    return count
