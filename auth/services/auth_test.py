from unittest.mock import AsyncMock
from uuid import uuid4

import bcrypt
import pytest

from auth.services.auth import AuthService  # путь к твоему файлу

JWT_SECRET = "secret"
JWT_ALGO = "HS256"
TOKEN_EXPIRE = 60


@pytest.mark.asyncio
async def test_register_user_success():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = None
    user_repo.upsert_user.return_value = {"id": 1, "username": "alice"}

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    user = await service.register_user("alice", "password123")

    assert user is not None
    assert user["username"] == "alice"
    user_repo.get_user_by_username.assert_awaited_once_with("alice")
    user_repo.upsert_user.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_user_existing():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {"id": 1, "username": "alice"}

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    user = await service.register_user("alice", "password123")

    assert user is None
    user_repo.get_user_by_username.assert_awaited_once_with("alice")
    user_repo.upsert_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticate_user_wrong_password():
    hashed_pw = b"$2b$12$KIX8c6dT7QsZJtZn3Nw7UuXUM4VhjQvhlFvPIoN2y2Z1jY6nChSeO"
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {
        "user_id": uuid4(),
        "password": hashed_pw.decode(),
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    token = await service.authenticate_user("alice", "wrongpass")

    assert token is None


@pytest.mark.asyncio
async def test_register_service_success():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = None
    service_repo.upsert_service.return_value = {"id": 1, "client_id": "service1"}

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    svc = await service.register_service("service1", "secret123")

    assert svc is not None
    assert svc["client_id"] == "service1"
    service_repo.get_service_by_client_id.assert_awaited_once_with("service1")
    service_repo.upsert_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_service_existing():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = {
        "id": 1,
        "client_id": "service1",
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    svc = await service.register_service("service1", "secret123")

    assert svc is None
    service_repo.upsert_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticate_user_success():
    password = "password123"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    # Устанавливаем, что get_user_by_username — это AsyncMock и возвращает нужный объект
    user_repo.get_user_by_username = AsyncMock(
        return_value={"user_id": uuid4(), "password": hashed_pw}
    )

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    token = await service.authenticate_user("alice", password)

    assert token is not None
    user_repo.get_user_by_username.assert_awaited_once_with("alice")


@pytest.mark.asyncio
async def test_authenticate_service_wrong_secret():
    hashed_secret = bcrypt.hashpw("secret123".encode(), bcrypt.gensalt()).decode()
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = {
        "service_id": uuid4(),
        "client_secret": hashed_secret,
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    token = await service.authenticate_service("service1", "wrongsecret")

    assert token is None
