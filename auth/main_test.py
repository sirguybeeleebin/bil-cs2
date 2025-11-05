# tests/test_auth.py
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import asyncpg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from auth.main import (
    POSTGRES_SCHEMA,
    AuthService,
    LoginServiceRequest,
    LoginServiceResponse,
    LoginUserRequest,
    LoginUserResponse,
    RegisterServiceRequest,
    RegisterServiceResponse,
    RegisterUserRequest,
    RegisterUserResponse,
    ServiceRepository,
    UserRepository,
    ph,
    router,
)


# -------------------------------
# Postgres test helper
# -------------------------------
async def init_postgres_container():
    container = PostgresContainer("postgres:15")
    container.start()
    dsn = container.get_connection_url().replace("+psycopg2", "")
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)

    async with pool.acquire() as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {POSTGRES_SCHEMA};")
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {POSTGRES_SCHEMA}.users (
                user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {POSTGRES_SCHEMA}.services (
                service_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                client_id TEXT UNIQUE NOT NULL,
                client_secret TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )
    return container, pool


# -------------------------------
# Repository tests
# -------------------------------
@pytest.mark.asyncio
async def test_user_repo_upsert_and_get():
    container, pool = await init_postgres_container()
    try:
        repo = UserRepository(pool, POSTGRES_SCHEMA)
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {POSTGRES_SCHEMA}.users CASCADE;")

        username = "alice"
        password_hash = "hashed_pw"

        user = await repo.upsert_user(username, password_hash)
        assert user["username"] == username
        assert UUID(str(user["user_id"]))

        fetched = await repo.get_by_username(username)
        assert fetched["username"] == username
        assert fetched["password_hash"] == password_hash
    finally:
        await pool.close()
        container.stop()


@pytest.mark.asyncio
async def test_service_repo_upsert_and_get():
    container, pool = await init_postgres_container()
    try:
        repo = ServiceRepository(pool, POSTGRES_SCHEMA)
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {POSTGRES_SCHEMA}.services CASCADE;")

        client_id = "svc1"
        client_secret = "secret"

        service = await repo.upsert_service(client_id, client_secret)
        assert service["client_id"] == client_id
        assert UUID(str(service["service_id"]))

        fetched = await repo.get_by_client_id(client_id)
        assert fetched["client_id"] == client_id
        assert fetched["client_secret"] == client_secret
    finally:
        await pool.close()
        container.stop()


# -------------------------------
# AuthService tests (mocks)
# -------------------------------
@pytest.mark.asyncio
async def test_auth_service_user_register_and_login():
    mock_user_repo = AsyncMock()
    auth_service = AuthService(mock_user_repo, None, "secret", "HS256", 60)

    username = "bob"
    password = "pw"

    # Register
    mock_user_repo.get_by_username.return_value = None
    mock_user_repo.upsert_user.return_value = {
        "user_id": str(uuid4()),
        "username": username,
    }
    result = await auth_service.register_user(username, password)
    assert "user_id" in result

    # Login
    hashed_pw = ph.hash(password)
    mock_user_repo.get_by_username.return_value = {
        "user_id": str(uuid4()),
        "username": username,
        "password_hash": hashed_pw,
    }
    token = await auth_service.login_user(username, password)
    assert "access_token" in token
    assert token["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_auth_service_service_register_and_login():
    mock_service_repo = AsyncMock()
    auth_service = AuthService(None, mock_service_repo, "secret", "HS256", 60)

    client_id = "svc2"
    client_secret = "sec"

    # Register
    mock_service_repo.get_by_client_id.return_value = None
    mock_service_repo.upsert_service.return_value = {
        "service_id": str(uuid4()),
        "client_id": client_id,
    }
    result = await auth_service.register_service(client_id, client_secret)
    assert "service_id" in result

    # Login
    mock_service_repo.get_by_client_id.return_value = {
        "service_id": str(uuid4()),
        "client_id": client_id,
        "client_secret": client_secret,
    }
    token = await auth_service.login_service(client_id, client_secret)
    assert "access_token" in token
    assert token["token_type"] == "bearer"


# -------------------------------
# Functional API tests (Pydantic)
# -------------------------------
def test_users_register_and_login_api():
    app = FastAPI()
    app.include_router(router)
    user_id = str(uuid4())
    app.state.auth_service = AsyncMock()
    app.state.auth_service.register_user.return_value = {"user_id": user_id}
    app.state.auth_service.login_user.return_value = {
        "access_token": "jwt_token",
        "token_type": "bearer",
    }

    client = TestClient(app)

    # Register
    req = RegisterUserRequest(username="alice", password="pw")
    resp = client.post("/api/v1/users/register", json=req.dict())
    assert resp.status_code == 200
    response_model = RegisterUserResponse(**resp.json())
    assert response_model.user_id == UUID(user_id)

    # Login
    req_login = LoginUserRequest(username="alice", password="pw")
    resp = client.post("/api/v1/users/login", json=req_login.dict())
    assert resp.status_code == 200
    token_model = LoginUserResponse(**resp.json())
    assert token_model.access_token == "jwt_token"
    assert token_model.token_type == "bearer"


def test_services_register_and_login_api():
    app = FastAPI()
    app.include_router(router)
    service_id = str(uuid4())
    app.state.auth_service = AsyncMock()
    app.state.auth_service.register_service.return_value = {"service_id": service_id}
    app.state.auth_service.login_service.return_value = {
        "access_token": "jwt_service_token",
        "token_type": "bearer",
    }

    client = TestClient(app)

    # Register
    req = RegisterServiceRequest(client_id="svc1", client_secret="sec")
    resp = client.post("/api/v1/services/register", json=req.dict())
    assert resp.status_code == 200
    response_model = RegisterServiceResponse(**resp.json())
    assert response_model.service_id == UUID(service_id)

    # Login
    req_login = LoginServiceRequest(client_id="svc1", client_secret="sec")
    resp = client.post("/api/v1/services/login", json=req_login.dict())
    assert resp.status_code == 200
    token_model = LoginServiceResponse(**resp.json())
    assert token_model.access_token == "jwt_service_token"
    assert token_model.token_type == "bearer"
