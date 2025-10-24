from unittest.mock import AsyncMock
from uuid import uuid4

import bcrypt
import pytest

from auth.services.auth import (
    AuthService,
    InvalidUserPasswordError,
    UserAlreadyExistsError,
)

JWT_SECRET = "secret"
JWT_ALGO = "HS256"
TOKEN_EXPIRE = 60


@pytest.mark.asyncio
async def test_register_user_success():
    user_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = None
    user_repo.upsert_user.return_value = {"user_id": uuid4(), "username": "alice"}

    service = AuthService(user_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    user = await service.register_user("alice", "Password123")

    assert user is not None
    assert user["username"] == "alice"
    user_repo.get_user_by_username.assert_awaited_once_with("alice")
    user_repo.upsert_user.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_user_existing():
    user_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {
        "user_id": uuid4(),
        "username": "alice",
    }

    service = AuthService(user_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(UserAlreadyExistsError):
        await service.register_user("alice", "Password123")


@pytest.mark.asyncio
async def test_authenticate_user_wrong_password():
    password = "Password123"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {
        "user_id": uuid4(),
        "password_hash": hashed_pw,
    }

    service = AuthService(user_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(InvalidUserPasswordError):
        await service.authenticate_user("alice", "WrongPass")


@pytest.mark.asyncio
async def test_authenticate_user_success():
    password = "Password123"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_repo = AsyncMock()
    user_repo.get_user_by_username.return_value = {
        "user_id": uuid4(),
        "password_hash": hashed_pw,
    }

    service = AuthService(user_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    token = await service.authenticate_user("alice", password)

    assert token is not None
    user_repo.get_user_by_username.assert_awaited_once_with("alice")
