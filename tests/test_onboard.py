"""Integration tests for the onboarding pipeline."""

import pytest

from helios.indexing.api_extractor import extract_api_surface
from helios.indexing.store import ContentStore


@pytest.fixture
async def store(tmp_path):
    s = ContentStore(data_dir=tmp_path)
    await s.initialize()
    yield s
    await s.close()


FASTAPI_CONTENT = """\
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

@router.get("/items")
async def list_items():
    return []

@router.post("/items")
async def create_item(body: CreateItemRequest):
    return body
"""


class TestOnboardPipeline:
    @pytest.mark.asyncio
    async def test_extract_and_store_endpoints(self, store):
        sid = await store.add_source("myproject", "/fake")
        await store.upsert_document(
            sid, "/fake/api.py", "api.py", FASTAPI_CONTENT, "python",
        )
        surface = await extract_api_surface(sid, store)
        endpoints = [ep.__dict__ for ep in surface.endpoints]
        count = await store.store_api_endpoints(sid, endpoints)
        assert count == 2

        stored = await store.get_api_endpoints(source_id=sid)
        assert len(stored) == 2
        methods = {e["method"] for e in stored}
        assert methods == {"GET", "POST"}

    @pytest.mark.asyncio
    async def test_endpoints_replaced_on_re_extract(self, store):
        sid = await store.add_source("myproject", "/fake")
        await store.upsert_document(
            sid, "/fake/api.py", "api.py", FASTAPI_CONTENT, "python",
        )

        # First extraction
        surface = await extract_api_surface(sid, store)
        await store.store_api_endpoints(sid, [ep.__dict__ for ep in surface.endpoints])
        assert len(await store.get_api_endpoints(source_id=sid)) == 2

        # Second extraction (same content — should replace, not duplicate)
        surface = await extract_api_surface(sid, store)
        await store.store_api_endpoints(sid, [ep.__dict__ for ep in surface.endpoints])
        assert len(await store.get_api_endpoints(source_id=sid)) == 2

    @pytest.mark.asyncio
    async def test_remove_source_clears_endpoints(self, store):
        sid = await store.add_source("myproject", "/fake")
        await store.upsert_document(
            sid, "/fake/api.py", "api.py", FASTAPI_CONTENT, "python",
        )
        surface = await extract_api_surface(sid, store)
        await store.store_api_endpoints(sid, [ep.__dict__ for ep in surface.endpoints])

        assert len(await store.get_api_endpoints(source_id=sid)) == 2
        await store.remove_source("myproject")
        assert len(await store.get_api_endpoints(source_id=sid)) == 0

    @pytest.mark.asyncio
    async def test_has_api_endpoints(self, store):
        sid = await store.add_source("myproject", "/fake")
        assert not await store.has_api_endpoints(sid)

        await store.upsert_document(
            sid, "/fake/api.py", "api.py", FASTAPI_CONTENT, "python",
        )
        surface = await extract_api_surface(sid, store)
        await store.store_api_endpoints(sid, [ep.__dict__ for ep in surface.endpoints])
        assert await store.has_api_endpoints(sid)
