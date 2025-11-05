# tests/main_test.py
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
    UserAlreadyExists,
    UserDoesNotExist,
    UserInvalidPassword,
    UserRepository,
    ph,
    router,
)




# -------------------------------
# Helper function to start container and create pool
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
                role TEXT NOT NULL,
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
async def test_upsert_and_get_user():
    container, pool = await init_postgres_container()
    try:
        user_repo = UserRepository(pool, schema=POSTGRES_SCHEMA)
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {POSTGRES_SCHEMA}.users CASCADE;")

        username = "alice"
        password_hash = "hashed_pw"
        role = "USER"

        user = await user_repo.upsert_user(username, password_hash, role)
        assert user["username"] == username
        assert UUID(str(user["user_id"]))

        fetched = await user_repo.get_by_username(username)
        assert fetched is not None
        assert fetched["username"] == username
        assert fetched["role"] == role
    finally:
        await pool.close()
        container.stop()


@pytest.mark.asyncio
async def test_get_nonexistent_user():
    container, pool = await init_postgres_container()
    try:
        user_repo = UserRepository(pool, schema=POSTGRES_SCHEMA)
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {POSTGRES_SCHEMA}.users CASCADE;")

        fetched = await user_repo.get_by_username("nonexist")
        assert fetched is None
    finally:
        await pool.close()
        container.stop()


@pytest.mark.asyncio
async def test_upsert_existing_user_updates():
    container, pool = await init_postgres_container()
    try:
        user_repo = UserRepository(pool, schema=POSTGRES_SCHEMA)
        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {POSTGRES_SCHEMA}.users CASCADE;")

        username = "bob"
        password_hash1 = "pw1"
        password_hash2 = "pw2"

        user1 = await user_repo.upsert_user(username, password_hash1, "USER")
        user2 = await user_repo.upsert_user(username, password_hash2, "ADMIN")

        assert user1["user_id"] == user2["user_id"]
        fetched = await user_repo.get_by_username(username)
        assert fetched["role"] == "ADMIN"
    finally:
        await pool.close()
        container.stop()


# -------------------------------
# Service tests with mocks
# -------------------------------
@pytest.mark.asyncio
async def test_auth_service_register_success():
    mock_repo = AsyncMock(spec=UserRepository)
    service = AuthService(mock_repo, "secret", "HS256", 60)

    username = "test_user"
    password = "Secret123!"
    role = "USER"

    mock_repo.get_by_username.return_value = None
    mock_repo.upsert_user.return_value = {
        "user_id": str(uuid4()),
        "username": username,
        "role": role,
    }

    result = await service.register(username, password, role)
    mock_repo.get_by_username.assert_awaited_with(username)
    mock_repo.upsert_user.assert_awaited()
    assert result["username"] == username


@pytest.mark.asyncio
async def test_auth_service_register_user_exists():
    mock_repo = AsyncMock(spec=UserRepository)
    service = AuthService(mock_repo, "secret", "HS256", 60)

    mock_repo.get_by_username.return_value = {"username": "exists"}
    with pytest.raises(UserAlreadyExists):
        await service.register("exists", "pw")


@pytest.mark.asyncio
async def test_auth_service_login_success():
    mock_repo = AsyncMock(spec=UserRepository)
    service = AuthService(mock_repo, "secret", "HS256", 60)

    password = "pw"
    hashed_pw = ph.hash(password)
    mock_repo.get_by_username.return_value = {
        "user_id": str(uuid4()),
        "username": "u",
        "password_hash": hashed_pw,
        "role": "USER",
    }

    token = await service.login("u", password)
    assert token is not None


@pytest.mark.asyncio
async def test_auth_service_login_wrong_password():
    mock_repo = AsyncMock(spec=UserRepository)
    service = AuthService(mock_repo, "secret", "HS256", 60)

    hashed_pw = ph.hash("correct")
    mock_repo.get_by_username.return_value = {
        "user_id": str(uuid4()),
        "username": "u",
        "password_hash": hashed_pw,
        "role": "USER",
    }

    with pytest.raises(UserInvalidPassword):
        await service.login("u", "wrong")


@pytest.mark.asyncio
async def test_auth_service_login_user_not_found():
    mock_repo = AsyncMock(spec=UserRepository)
    service = AuthService(mock_repo, "secret", "HS256", 60)

    mock_repo.get_by_username.return_value = None
    with pytest.raises(UserDoesNotExist):
        await service.login("no_user", "any")


# -------------------------------
# Functional API tests (synchronous)
# -------------------------------
def test_register_user():
    app = FastAPI()
    app.include_router(router)

    user_id = str(uuid4())
    app.state.auth_service = AsyncMock()
    app.state.auth_service.register.return_value = {"user_id": user_id}

    client = TestClient(app)
    response = client.post(
        "/api/v1/register", json={"username": "alice", "password": "pw", "role": "USER"}
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == user_id


def test_register_user_exists():
    app = FastAPI()
    app.include_router(router)

    app.state.auth_service = AsyncMock()
    app.state.auth_service.register.side_effect = UserAlreadyExists()

    client = TestClient(app)
    response = client.post(
        "/api/v1/register", json={"username": "alice", "password": "pw", "role": "USER"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Пользователь уже существует"


def test_login_user_success():
    app = FastAPI()
    app.include_router(router)

    token = "jwt_token"
    app.state.auth_service = AsyncMock()
    app.state.auth_service.login.return_value = token

    client = TestClient(app)
    response = client.post(
        "/api/v1/login", json={"username": "alice", "password": "pw"}
    )

    assert response.status_code == 200
    assert response.json()["access_token"] == token


def test_login_user_not_found():
    app = FastAPI()
    app.include_router(router)

    app.state.auth_service = AsyncMock()
    app.state.auth_service.login.side_effect = UserDoesNotExist()

    client = TestClient(app)
    response = client.post(
        "/api/v1/login", json={"username": "alice", "password": "pw"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Пользователь не найден"


def test_login_user_wrong_password():
    app = FastAPI()
    app.include_router(router)

    app.state.auth_service = AsyncMock()
    app.state.auth_service.login.side_effect = UserInvalidPassword()

    client = TestClient(app)
    response = client.post(
        "/api/v1/login", json={"username": "alice", "password": "pw"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Неверный пароль"
