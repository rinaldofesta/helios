"""HuggingFace Hub integration — search, download, and manage GGUF models.

Provides LM Studio–style model management: browse HuggingFace for GGUF
quantisations, download them to a local directory, and list locally
available models.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path.home() / ".helios" / "models"


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #


@dataclass
class HubModel:
    """A model repository on HuggingFace Hub."""

    repo_id: str
    name: str
    author: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    gguf_files: list[GGUFFile] = field(default_factory=list)


@dataclass
class GGUFFile:
    """A GGUF file inside a HuggingFace repo."""

    filename: str
    size_bytes: int
    repo_id: str
    quantisation: str = ""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    @property
    def size_display(self) -> str:
        if self.size_bytes >= 1024**3:
            return f"{self.size_gb:.1f} GB"
        return f"{self.size_bytes / (1024**2):.0f} MB"


@dataclass
class LocalModel:
    """A locally downloaded model file."""

    path: Path
    filename: str
    repo_id: str
    size_bytes: int

    @property
    def size_display(self) -> str:
        gb = self.size_bytes / (1024**3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        return f"{self.size_bytes / (1024**2):.0f} MB"


@dataclass
class DownloadProgress:
    """Progress info for an active download."""

    filename: str
    repo_id: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    status: str = "pending"  # pending, downloading, complete, error
    error: str | None = None

    @property
    def percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100


# --------------------------------------------------------------------------- #
#  Hub client
# --------------------------------------------------------------------------- #


class ModelHub:
    """Search HuggingFace Hub for GGUF models and manage local downloads."""

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self._models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._downloads: dict[str, DownloadProgress] = {}

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    # ---- Search ----------------------------------------------------------- #

    async def search(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
    ) -> list[HubModel]:
        """Search HuggingFace Hub for GGUF-compatible models."""
        return await asyncio.to_thread(self._search_sync, query, limit, sort)

    def _search_sync(self, query: str, limit: int, sort: str = "downloads") -> list[HubModel]:
        from huggingface_hub import HfApi

        api = HfApi()
        results: list[HubModel] = []

        # Search for models tagged with "gguf"
        models = api.list_models(
            search=query,
            filter="gguf",
            sort=sort,
            limit=limit,
        )

        for m in models:
            author, _, name = (m.id or "").partition("/")
            results.append(HubModel(
                repo_id=m.id or "",
                name=name or m.id or "",
                author=author,
                downloads=m.downloads or 0,
                likes=m.likes or 0,
                tags=list(m.tags or []),
            ))

        return results

    # ---- List GGUF files in a repo ---------------------------------------- #

    async def list_gguf_files(self, repo_id: str) -> list[GGUFFile]:
        """List all GGUF files available in a HuggingFace repo."""
        return await asyncio.to_thread(self._list_gguf_files_sync, repo_id)

    def _list_gguf_files_sync(self, repo_id: str) -> list[GGUFFile]:
        from huggingface_hub import HfApi

        api = HfApi()
        files: list[GGUFFile] = []

        try:
            siblings = api.model_info(repo_id, files_metadata=True).siblings or []
        except Exception as e:
            logger.warning("Failed to get model info for %s: %s", repo_id, e)
            return []

        for sibling in siblings:
            fname = sibling.rfilename
            if not fname.lower().endswith(".gguf"):
                continue

            # Guess quantisation from filename (e.g., Q4_K_M, Q5_K_S, etc.)
            quant = _guess_quantisation(fname)

            files.append(GGUFFile(
                filename=fname,
                size_bytes=sibling.size or 0,
                repo_id=repo_id,
                quantisation=quant,
            ))

        # Sort by size ascending
        files.sort(key=lambda f: f.size_bytes)
        return files

    # ---- Download --------------------------------------------------------- #

    async def download(
        self,
        repo_id: str,
        filename: str,
        on_progress: callable | None = None,
    ) -> Path:
        """Download a GGUF file from HuggingFace Hub.

        Returns the local path to the downloaded file.
        """
        progress_key = f"{repo_id}/{filename}"
        self._downloads[progress_key] = DownloadProgress(
            filename=filename,
            repo_id=repo_id,
            status="downloading",
        )

        try:
            local_path = await asyncio.to_thread(
                self._download_sync, repo_id, filename, progress_key, on_progress
            )
            self._downloads[progress_key].status = "complete"
            return local_path
        except Exception as e:
            self._downloads[progress_key].status = "error"
            self._downloads[progress_key].error = str(e)
            raise

    def _download_sync(
        self,
        repo_id: str,
        filename: str,
        progress_key: str,
        on_progress: callable | None,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        # Download to our models directory, organized by repo
        repo_dir = self._models_dir / repo_id.replace("/", "--")
        repo_dir.mkdir(parents=True, exist_ok=True)

        local_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(repo_dir),
        ))

        return local_path

    # ---- Local model management ------------------------------------------- #

    def list_local_models(self) -> list[LocalModel]:
        """List all locally downloaded GGUF models."""
        models: list[LocalModel] = []

        if not self._models_dir.exists():
            return models

        for gguf_file in self._models_dir.rglob("*.gguf"):
            # Reconstruct repo_id from directory name
            relative = gguf_file.relative_to(self._models_dir)
            parts = list(relative.parts)
            if len(parts) >= 2:
                repo_id = parts[0].replace("--", "/")
            else:
                repo_id = "local"

            models.append(LocalModel(
                path=gguf_file,
                filename=gguf_file.name,
                repo_id=repo_id,
                size_bytes=gguf_file.stat().st_size,
            ))

        models.sort(key=lambda m: m.filename)
        return models

    def delete_local_model(self, path: str | Path) -> bool:
        """Delete a locally downloaded GGUF file."""
        p = Path(path)
        if p.exists() and p.suffix == ".gguf" and self._models_dir in p.parents:
            p.unlink()
            # Clean up empty parent dirs
            parent = p.parent
            if parent != self._models_dir and not any(parent.iterdir()):
                parent.rmdir()
            return True
        return False

    def get_download_progress(self) -> dict[str, DownloadProgress]:
        """Get progress for all active/recent downloads."""
        return dict(self._downloads)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _guess_quantisation(filename: str) -> str:
    """Guess the quantisation level from a GGUF filename."""
    import re

    # Common patterns: Q4_K_M, Q5_K_S, Q8_0, IQ4_XS, etc.
    match = re.search(r"[.-]((?:I?Q\d+_[A-Z0-9_]+)|(?:f(?:16|32)))", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: look for Q followed by a digit
    match = re.search(r"[.-](Q\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""
