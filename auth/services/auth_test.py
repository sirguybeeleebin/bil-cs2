from unittest.mock import AsyncMock
from uuid import uuid4

import bcrypt
import pytest

from auth.services.auth import (
    AuthService,
    InvalidServiceSecretError,
    InvalidUserPasswordError,
    ServiceAlreadyExistsError,
    UserAlreadyExistsError,
)

JWT_SECRET = "secret"
JWT_ALGO = "HS256"
TOKEN_EXPIRE = 60


@pytest.mark.asyncio
async def test_register_user_success():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = None
    user_repo.upsert_user.return_value = {"user_id": uuid4(), "username": "alice"}

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    user = await service.register_user("alice", "Password123")

    assert user is not None
    assert user["username"] == "alice"
    user_repo.get_user_by_username.assert_awaited_once_with("alice")
    user_repo.upsert_user.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_user_existing():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {
        "user_id": uuid4(),
        "username": "alice",
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(UserAlreadyExistsError):
        await service.register_user("alice", "Password123")


@pytest.mark.asyncio
async def test_authenticate_user_wrong_password():
    password = "Password123"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {
        "user_id": uuid4(),
        "password_hash": hashed_pw,
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(InvalidUserPasswordError):
        await service.authenticate_user("alice", "WrongPass")


@pytest.mark.asyncio
async def test_authenticate_user_success():
    password = "Password123"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {
        "user_id": uuid4(),
        "password_hash": hashed_pw,
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    token = await service.authenticate_user("alice", password)

    assert token is not None
    user_repo.get_user_by_username.assert_awaited_once_with("alice")


@pytest.mark.asyncio
async def test_register_service_success():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = None
    service_repo.upsert_service.return_value = {
        "service_id": uuid4(),
        "client_id": "service1",
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    svc = await service.register_service("service1", "Secret123")

    assert svc is not None
    assert svc["client_id"] == "service1"
    service_repo.get_service_by_client_id.assert_awaited_once_with("service1")
    service_repo.upsert_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_service_existing():
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = {
        "service_id": uuid4(),
        "client_id": "service1",
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(ServiceAlreadyExistsError):
        await service.register_service("service1", "Secret123")


@pytest.mark.asyncio
async def test_authenticate_service_wrong_secret():
    hashed_secret = bcrypt.hashpw("Secret123".encode(), bcrypt.gensalt()).decode()
    user_repo = AsyncMock()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = {
        "service_id": uuid4(),
        "client_secret_hash": hashed_secret,
    }

    service = AuthService(user_repo, service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(InvalidServiceSecretError):
        await service.authenticate_service("service1", "WrongSecret")
