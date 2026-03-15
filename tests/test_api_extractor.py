"""Tests for API surface extraction from indexed documents."""

import pytest

from helios.indexing.api_extractor import (
    _extract_express,
    _extract_fastapi,
    _extract_nestjs,
    _extract_nextjs,
    _nextjs_path_from_file,
    extract_api_surface,
)


# --------------------------------------------------------------------------- #
#  Next.js App Router
# --------------------------------------------------------------------------- #


class TestNextJsPathFromFile:
    def test_simple_route(self):
        assert _nextjs_path_from_file("app/api/users/route.ts") == "/api/users"

    def test_dynamic_segment(self):
        assert _nextjs_path_from_file("app/api/users/[id]/route.ts") == "/api/users/:id"

    def test_catch_all(self):
        assert _nextjs_path_from_file("app/api/auth/[...nextauth]/route.ts") == "/api/auth/:nextauth*"

    def test_nested_dynamic(self):
        assert _nextjs_path_from_file("app/api/orgs/[orgId]/members/[memberId]/route.ts") == "/api/orgs/:orgId/members/:memberId"

    def test_non_route_file(self):
        assert _nextjs_path_from_file("app/api/users/service.ts") == ""

    def test_jsx_extension(self):
        assert _nextjs_path_from_file("app/api/users/route.jsx") == "/api/users"

    def test_src_prefix(self):
        assert _nextjs_path_from_file("src/app/api/users/route.ts") == "/api/users"


NEXTJS_ROUTE_CONTENT = """\
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const users = await db.users.findMany();
  return NextResponse.json(users);
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const user = await db.users.create({ data: body });
  return NextResponse.json(user, { status: 201 });
}
"""

NEXTJS_CONST_EXPORT = """\
import { NextResponse } from 'next/server';

export const GET = async () => {
  return NextResponse.json({ status: 'ok' });
}

export const DELETE = async (req: Request) => {
  return NextResponse.json({ deleted: true });
}
"""


class TestNextJsExtraction:
    def test_exported_functions(self):
        eps = _extract_nextjs(NEXTJS_ROUTE_CONTENT, "app/api/users/route.ts")
        assert len(eps) == 2
        assert eps[0].method == "GET"
        assert eps[0].path == "/api/users"
        assert eps[0].framework == "nextjs"
        assert eps[1].method == "POST"

    def test_const_exports(self):
        eps = _extract_nextjs(NEXTJS_CONST_EXPORT, "app/api/health/route.ts")
        assert len(eps) == 2
        methods = {e.method for e in eps}
        assert methods == {"GET", "DELETE"}

    def test_dynamic_route(self):
        eps = _extract_nextjs(
            "export async function GET() { return Response.json({}); }",
            "app/api/users/[id]/route.ts",
        )
        assert len(eps) == 1
        assert eps[0].path == "/api/users/:id"

    def test_no_exports(self):
        eps = _extract_nextjs("const handler = () => {};", "app/api/users/route.ts")
        assert len(eps) == 0


# --------------------------------------------------------------------------- #
#  FastAPI
# --------------------------------------------------------------------------- #

FASTAPI_CONTENT = """\
from fastapi import APIRouter
from pydantic import BaseModel

class CreateUserRequest(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str

router = APIRouter(prefix="/api/v1")

@router.get("/users", response_model=UserResponse)
async def list_users():
    return await User.all()

@router.post("/users")
async def create_user(body: CreateUserRequest) -> UserResponse:
    return await User.create(**body.dict())

@router.delete("/users/{user_id}")
async def delete_user(user_id: int):
    await User.delete(user_id)
"""


class TestFastApiExtraction:
    def test_methods_and_paths(self):
        eps = _extract_fastapi(FASTAPI_CONTENT, "api/routes/users.py")
        assert len(eps) == 3
        assert eps[0].method == "GET"
        assert eps[0].path == "/api/v1/users"
        assert eps[1].method == "POST"
        assert eps[2].method == "DELETE"

    def test_handler_names(self):
        eps = _extract_fastapi(FASTAPI_CONTENT, "api/routes/users.py")
        assert eps[0].handler == "list_users"
        assert eps[1].handler == "create_user"
        assert eps[2].handler == "delete_user"

    def test_response_model(self):
        eps = _extract_fastapi(FASTAPI_CONTENT, "api/routes/users.py")
        assert eps[0].response_type == "UserResponse"

    def test_request_body_type(self):
        eps = _extract_fastapi(FASTAPI_CONTENT, "api/routes/users.py")
        assert eps[1].request_type == "CreateUserRequest"

    def test_framework_tag(self):
        eps = _extract_fastapi(FASTAPI_CONTENT, "api/routes/users.py")
        assert all(e.framework == "fastapi" for e in eps)

    def test_no_prefix(self):
        content = '@router.get("/health")\nasync def healthcheck():\n    return {"ok": True}'
        eps = _extract_fastapi(content, "main.py")
        assert len(eps) == 1
        assert eps[0].path == "/health"


# --------------------------------------------------------------------------- #
#  NestJS
# --------------------------------------------------------------------------- #

NESTJS_CONTENT = """\
import { Controller, Get, Post, Body, Param } from '@nestjs/common';
import { CreateUserDto } from './dto/create-user.dto';
import { UsersService } from './users.service';

@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Get()
  async findAll() {
    return this.usersService.findAll();
  }

  @Get(':id')
  async findOne(@Param('id') id: string) {
    return this.usersService.findOne(id);
  }

  @Post()
  async create(@Body() createUserDto: CreateUserDto) {
    return this.usersService.create(createUserDto);
  }
}
"""


class TestNestJsExtraction:
    def test_controller_base_path(self):
        eps = _extract_nestjs(NESTJS_CONTENT, "src/users/users.controller.ts")
        assert all(e.path.startswith("/users") for e in eps)

    def test_method_count(self):
        eps = _extract_nestjs(NESTJS_CONTENT, "src/users/users.controller.ts")
        assert len(eps) == 3

    def test_combined_paths(self):
        eps = _extract_nestjs(NESTJS_CONTENT, "src/users/users.controller.ts")
        paths = {e.path for e in eps}
        assert "/users" in paths
        assert "/users/:id" in paths

    def test_handler_names(self):
        eps = _extract_nestjs(NESTJS_CONTENT, "src/users/users.controller.ts")
        handlers = {e.handler for e in eps}
        assert "findAll" in handlers
        assert "findOne" in handlers
        assert "create" in handlers

    def test_body_dto(self):
        eps = _extract_nestjs(NESTJS_CONTENT, "src/users/users.controller.ts")
        post = next(e for e in eps if e.method == "POST")
        assert post.request_type == "CreateUserDto"

    def test_framework_tag(self):
        eps = _extract_nestjs(NESTJS_CONTENT, "src/users/users.controller.ts")
        assert all(e.framework == "nestjs" for e in eps)


# --------------------------------------------------------------------------- #
#  Express
# --------------------------------------------------------------------------- #

EXPRESS_CONTENT = """\
const express = require('express');
const router = express.Router();

router.get('/users', getUsers);
router.post('/users', createUser);
router.get('/users/:id', getUserById);
router.delete('/users/:id', deleteUser);
"""


class TestExpressExtraction:
    def test_route_count(self):
        eps = _extract_express(EXPRESS_CONTENT, "routes/users.js")
        assert len(eps) == 4

    def test_methods(self):
        eps = _extract_express(EXPRESS_CONTENT, "routes/users.js")
        methods = [e.method for e in eps]
        assert methods == ["GET", "POST", "GET", "DELETE"]

    def test_paths(self):
        eps = _extract_express(EXPRESS_CONTENT, "routes/users.js")
        paths = [e.path for e in eps]
        assert paths == ["/users", "/users", "/users/:id", "/users/:id"]

    def test_handler_names(self):
        eps = _extract_express(EXPRESS_CONTENT, "routes/users.js")
        assert eps[0].handler == "getUsers"
        assert eps[2].handler == "getUserById"

    def test_framework_tag(self):
        eps = _extract_express(EXPRESS_CONTENT, "routes/users.js")
        assert all(e.framework == "express" for e in eps)

    def test_anonymous_handler(self):
        content = "app.get('/health', (req, res) => res.send('ok'))"
        eps = _extract_express(content, "app.js")
        assert len(eps) == 1
        assert eps[0].handler == "anonymous"


# --------------------------------------------------------------------------- #
#  Integration with store
# --------------------------------------------------------------------------- #


@pytest.fixture
async def store(tmp_path):
    from helios.indexing.store import ContentStore

    s = ContentStore(data_dir=tmp_path)
    await s.initialize()
    yield s
    await s.close()


class TestExtractApiSurface:
    @pytest.mark.asyncio
    async def test_nextjs_from_store(self, store):
        sid = await store.add_source("myproject", "/fake/path")
        await store.upsert_document(
            sid, "/fake/path/app/api/users/route.ts", "app/api/users/route.ts",
            NEXTJS_ROUTE_CONTENT, "typescript",
        )
        surface = await extract_api_surface(sid, store)
        assert len(surface.endpoints) == 2
        assert "nextjs" in surface.frameworks

    @pytest.mark.asyncio
    async def test_fastapi_from_store(self, store):
        sid = await store.add_source("backend", "/fake/path")
        await store.upsert_document(
            sid, "/fake/path/api/users.py", "api/users.py",
            FASTAPI_CONTENT, "python",
        )
        surface = await extract_api_surface(sid, store)
        assert len(surface.endpoints) == 3
        assert "fastapi" in surface.frameworks

    @pytest.mark.asyncio
    async def test_empty_project(self, store):
        sid = await store.add_source("empty", "/fake/path")
        surface = await extract_api_surface(sid, store)
        assert len(surface.endpoints) == 0
        assert len(surface.frameworks) == 0

    @pytest.mark.asyncio
    async def test_mixed_frameworks(self, store):
        sid = await store.add_source("fullstack", "/fake/path")
        await store.upsert_document(
            sid, "/fake/path/app/api/users/route.ts", "app/api/users/route.ts",
            NEXTJS_ROUTE_CONTENT, "typescript",
        )
        await store.upsert_document(
            sid, "/fake/path/backend/users.py", "backend/users.py",
            FASTAPI_CONTENT, "python",
        )
        surface = await extract_api_surface(sid, store)
        assert len(surface.endpoints) == 5
        assert "nextjs" in surface.frameworks
        assert "fastapi" in surface.frameworks
