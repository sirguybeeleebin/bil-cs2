import os
from uuid import UUID

import asyncpg
import httpx
import pytest
import pytest_asyncio
from argon2 import PasswordHasher

API_URL = "http://auth:8000/api/v1"

# Database settings
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_USER = os.getenv("POSTGRES_USER", "cs2_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "cs2_password")
DB_NAME = os.getenv("POSTGRES_DB", "cs2_db")
DB_SCHEMA = "auth"  # <-- добавлена схема

ph = PasswordHasher()


# Function-scoped HTTP client
@pytest_asyncio.fixture
async def async_client():
    async with httpx.AsyncClient(base_url=API_URL) as client:
        yield client


# Fixture to create and truncate users table in schema 'auth'
@pytest_asyncio.fixture
async def setup_users_table():
    conn = await asyncpg.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

    # Создаём схему auth, если не существует
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};")

    # Создаём расширение uuid-ossp
    await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

    # Создаём таблицу users в схеме auth
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.users (
            user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'USER',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    """)

    # Создаём уникальный индекс на username
    await conn.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON {DB_SCHEMA}.users(username);
    """)

    # Очистка таблицы перед тестами
    await conn.execute(f"TRUNCATE TABLE {DB_SCHEMA}.users RESTART IDENTITY CASCADE;")

    yield  # run the test

    # Очистка таблицы после тестов
    await conn.execute(f"TRUNCATE TABLE {DB_SCHEMA}.users RESTART IDENTITY CASCADE;")
    await conn.close()


@pytest.mark.asyncio
async def test_register_user(async_client, setup_users_table):
    username = "functional_user"
    password = "Secret123!"
    role = "USER"

    # Успешная регистрация
    resp = await async_client.post(
        "/register", json={"username": username, "password": password, "role": role}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "user_id" in data
    UUID(data["user_id"])

    # Попытка зарегистрировать существующего пользователя
    resp2 = await async_client.post(
        "/register", json={"username": username, "password": password, "role": role}
    )
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "Пользователь уже существует"


@pytest.mark.asyncio
async def test_login_user(async_client, setup_users_table):
    username = "functional_user"
    password = "Secret123!"
    role = "USER"

    # Убедимся, что пользователь существует
    await async_client.post(
        "/register", json={"username": username, "password": password, "role": role}
    )

    # Успешный логин
    resp = await async_client.post(
        "/login", json={"username": username, "password": password}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    # Логин с неверным паролем
    resp2 = await async_client.post(
        "/login", json={"username": username, "password": "wrong_password"}
    )
    assert resp2.status_code == 401
    assert resp2.json()["detail"] == "Неверный пароль"

    # Логин несуществующего пользователя
    resp3 = await async_client.post(
        "/login", json={"username": "no_such_user", "password": "any"}
    )
    assert resp3.status_code == 404
    assert resp3.json()["detail"] == "Пользователь не найден"
