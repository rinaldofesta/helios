"""Tests for project stack analysis."""

import json

import pytest

from helios.indexing.stack_analyzer import (
    analyze_stack,
    detect_api_clients,
    detect_fe_libraries,
    detect_project_patterns,
    extract_type_defs,
)
from helios.indexing.store import ContentStore


# --------------------------------------------------------------------------- #
#  FE Library Detection
# --------------------------------------------------------------------------- #


class TestDetectFeLibraries:
    def test_detects_react_and_next(self, tmp_path):
        pkg = {
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "next": "14.1.0",
            },
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        libs = detect_fe_libraries(tmp_path)
        names = {lib.name for lib in libs}
        assert "react" in names
        assert "next" in names

    def test_detects_tailwind_and_shadcn(self, tmp_path):
        pkg = {
            "dependencies": {
                "react": "^18.2.0",
                "tailwindcss": "^3.4.0",
                "class-variance-authority": "^0.7.0",
                "@radix-ui/react-slot": "^1.0.0",
            },
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "components.json").write_text("{}")
        libs = detect_fe_libraries(tmp_path)
        names = {lib.name for lib in libs}
        assert "tailwindcss" in names
        assert "shadcn/ui" in names

    def test_detects_state_management(self, tmp_path):
        pkg = {"dependencies": {"zustand": "^4.5.0", "react": "^18.0.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        libs = detect_fe_libraries(tmp_path)
        zustand = next(lib for lib in libs if lib.name == "zustand")
        assert zustand.category == "state_management"

    def test_detects_data_fetching(self, tmp_path):
        pkg = {"dependencies": {"@tanstack/react-query": "^5.0.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        libs = detect_fe_libraries(tmp_path)
        assert any(lib.name == "@tanstack/react-query" for lib in libs)

    def test_no_package_json(self, tmp_path):
        libs = detect_fe_libraries(tmp_path)
        assert libs == []

    def test_categories_are_set(self, tmp_path):
        pkg = {
            "dependencies": {
                "react": "^18.0.0",
                "zustand": "^4.0.0",
                "tailwindcss": "^3.0.0",
                "framer-motion": "^11.0.0",
            },
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        libs = detect_fe_libraries(tmp_path)
        cats = {lib.category for lib in libs}
        assert "ui_framework" in cats
        assert "state_management" in cats
        assert "css" in cats
        assert "animation" in cats


# --------------------------------------------------------------------------- #
#  Type/Schema Extraction
# --------------------------------------------------------------------------- #


@pytest.fixture
async def store(tmp_path):
    s = ContentStore(data_dir=tmp_path)
    await s.initialize()
    yield s
    await s.close()


TS_TYPES_CONTENT = """\
export interface UserDto {
  id: string;
  name: string;
  email: string;
}

export interface CreateUserRequest {
  name: string;
  email: string;
}

export type UserResponse = {
  user: UserDto;
  token: string;
}

const internalHelper = () => {};
"""

ZOD_CONTENT = """\
import { z } from 'zod';

export const createUserSchema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
});

export const updateUserSchema = z.object({
  name: z.string().optional(),
});
"""

PYDANTIC_CONTENT = """\
from pydantic import BaseModel

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class CreateOrderRequest(BaseModel):
    product_id: int
    quantity: int
"""


class TestExtractTypeDefs:
    @pytest.mark.asyncio
    async def test_typescript_interfaces(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/types.ts", "types.ts", TS_TYPES_CONTENT, "typescript")
        types = await extract_type_defs(sid, store)
        names = {t.name for t in types}
        assert "UserDto" in names
        assert "CreateUserRequest" in names
        assert "UserResponse" in names

    @pytest.mark.asyncio
    async def test_zod_schemas(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/schemas.ts", "schemas.ts", ZOD_CONTENT, "typescript")
        types = await extract_type_defs(sid, store)
        names = {t.name for t in types}
        assert "createUserSchema" in names
        assert "updateUserSchema" in names
        assert all(t.kind == "zod_schema" for t in types)

    @pytest.mark.asyncio
    async def test_pydantic_models(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/models.py", "models.py", PYDANTIC_CONTENT, "python")
        types = await extract_type_defs(sid, store)
        names = {t.name for t in types}
        assert "UserResponse" in names
        assert "CreateOrderRequest" in names

    @pytest.mark.asyncio
    async def test_api_types_sorted_first(self, store):
        sid = await store.add_source("proj", "/fake")
        content = TS_TYPES_CONTENT + "\nexport interface AppConfig {\n  debug: boolean;\n}\n"
        await store.upsert_document(sid, "/fake/types.ts", "types.ts", content, "typescript")
        types = await extract_type_defs(sid, store)
        # API-related types (Dto/Request/Response) should come before others
        names = [t.name for t in types]
        # AppConfig has no API hint, should be last
        assert names.index("AppConfig") > names.index("CreateUserRequest")

    @pytest.mark.asyncio
    async def test_preview_content(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/types.ts", "types.ts", TS_TYPES_CONTENT, "typescript")
        types = await extract_type_defs(sid, store)
        user_dto = next(t for t in types if t.name == "UserDto")
        assert "id: string" in user_dto.preview


# --------------------------------------------------------------------------- #
#  API Client Detection
# --------------------------------------------------------------------------- #

AXIOS_CLIENT = """\
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: '/api',
  timeout: 5000,
});
"""

TRPC_CLIENT = """\
import { createTRPCReact } from '@trpc/react-query';
import type { AppRouter } from '../server/trpc';

export const trpc = createTRPCReact<AppRouter>();
"""

REACT_QUERY_HOOKS = """\
import { useQuery, useMutation } from '@tanstack/react-query';

export function useUsersQuery() {
  return useQuery({ queryKey: ['users'], queryFn: fetchUsers });
}

export function useCreateUserMutation() {
  return useMutation({ mutationFn: createUser });
}
"""


class TestDetectApiClients:
    @pytest.mark.asyncio
    async def test_axios_client(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/lib/api.ts", "lib/api.ts", AXIOS_CLIENT, "typescript")
        clients = await detect_api_clients(sid, store)
        assert any(c.kind == "axios" for c in clients)

    @pytest.mark.asyncio
    async def test_trpc_client(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/lib/trpc.ts", "lib/trpc.ts", TRPC_CLIENT, "typescript")
        clients = await detect_api_clients(sid, store)
        assert any(c.kind == "trpc" for c in clients)

    @pytest.mark.asyncio
    async def test_react_query_hooks(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/hooks/users.ts", "hooks/users.ts", REACT_QUERY_HOOKS, "typescript")
        clients = await detect_api_clients(sid, store)
        assert any(c.kind == "react_query_hooks" for c in clients)
        hook_client = next(c for c in clients if c.kind == "react_query_hooks")
        assert "useUsersQuery" in hook_client.description

    @pytest.mark.asyncio
    async def test_api_service_file(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(
            sid, "/fake/services/api.ts", "services/api.ts",
            "export const baseUrl = '/api/v1';", "typescript",
        )
        clients = await detect_api_clients(sid, store)
        assert any(c.kind == "api_service" for c in clients)


# --------------------------------------------------------------------------- #
#  Project Pattern Detection
# --------------------------------------------------------------------------- #


class TestDetectProjectPatterns:
    @pytest.mark.asyncio
    async def test_detects_folders(self, store):
        sid = await store.add_source("proj", "/fake")
        await store.upsert_document(sid, "/fake/components/Button.tsx", "components/Button.tsx", "export const Button = () => <button/>;", "typescript")
        await store.upsert_document(sid, "/fake/hooks/useAuth.ts", "hooks/useAuth.ts", "export function useAuth() {}", "typescript")
        await store.upsert_document(sid, "/fake/lib/utils.ts", "lib/utils.ts", "export const cn = () => {};", "typescript")
        patterns = await detect_project_patterns(sid, store)
        assert "components" in patterns.folder_structure
        assert "hooks" in patterns.folder_structure
        assert "lib" in patterns.folder_structure

    @pytest.mark.asyncio
    async def test_detects_pascal_case(self, store):
        sid = await store.add_source("proj", "/fake")
        for name in ["Button", "Header", "UserProfile", "SideBar"]:
            await store.upsert_document(
                sid, f"/fake/components/{name}.tsx", f"components/{name}.tsx",
                f"export const {name} = () => <div/>;", "typescript",
            )
        patterns = await detect_project_patterns(sid, store)
        assert patterns.naming_convention == "PascalCase"

    @pytest.mark.asyncio
    async def test_detects_component_style(self, store):
        sid = await store.add_source("proj", "/fake")
        for name in ["Button", "Header", "Card"]:
            await store.upsert_document(
                sid, f"/fake/{name}.tsx", f"{name}.tsx",
                f"export const {name} = () => <div/>;", "typescript",
            )
        patterns = await detect_project_patterns(sid, store)
        assert "named exports" in patterns.component_style


# --------------------------------------------------------------------------- #
#  Full Analysis
# --------------------------------------------------------------------------- #


class TestAnalyzeStack:
    @pytest.mark.asyncio
    async def test_full_analysis(self, store, tmp_path):
        # Create package.json
        pkg = {"dependencies": {"react": "^18.0.0", "tailwindcss": "^3.0.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))

        sid = await store.add_source("proj", str(tmp_path))
        await store.upsert_document(
            sid, str(tmp_path / "types.ts"), "types.ts",
            TS_TYPES_CONTENT, "typescript",
        )
        await store.upsert_document(
            sid, str(tmp_path / "lib/api.ts"), "lib/api.ts",
            AXIOS_CLIENT, "typescript",
        )

        analysis = await analyze_stack(sid, tmp_path, store)
        assert len(analysis.libraries) > 0
        assert len(analysis.type_defs) > 0
        assert len(analysis.api_clients) > 0
        assert analysis.patterns is not None
