from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import bcrypt
import jwt
import pytest

from auth.services.auth import AuthService


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def mock_user_repo():
    repo = AsyncMock()
    return repo


@pytest.fixture
def auth_service(mock_user_repo):
    return AuthService(
        user_repo=mock_user_repo,
        jwt_secret="testsecret",
        jwt_algorithm="HS256",
        access_token_expire_minutes=60,
    )


# -----------------------------
# Tests
# -----------------------------
@pytest.mark.asyncio
async def test_register_success(auth_service, mock_user_repo):
    # No existing user
    mock_user_repo.get_by_username.return_value = None
    mock_user_repo.upsert.return_value = {"user_id": 1, "username": "alice"}

    with (
        patch("bcrypt.hashpw", return_value=b"hashed_pw") as mock_hash,
        patch("jwt.encode", return_value="token123") as mock_jwt,
    ):
        token = await auth_service.register("alice", "password")

    mock_user_repo.get_by_username.assert_awaited_once_with("alice")
    mock_user_repo.upsert.assert_awaited_once()
    mock_hash.assert_called_once()
    mock_jwt.assert_called_once()
    assert token == "token123"


@pytest.mark.asyncio
async def test_register_existing_user_raises(auth_service, mock_user_repo):
    mock_user_repo.get_by_username.return_value = {"user_id": 1, "username": "alice"}

    with pytest.raises(ValueError, match="Username already exists"):
        await auth_service.register("alice", "password")


@pytest.mark.asyncio
async def test_login_success(auth_service, mock_user_repo):
    hashed_pw = bcrypt.hashpw("password".encode(), bcrypt.gensalt())
    mock_user_repo.get_by_username.return_value = {
        "user_id": 1,
        "username": "alice",
        "password_hash": hashed_pw.decode(),
    }

    with patch("jwt.encode", return_value="token123") as mock_jwt:
        token = await auth_service.login("alice", "password")

    mock_user_repo.get_by_username.assert_awaited_once_with("alice")
    mock_jwt.assert_called_once()
    assert token == "token123"


@pytest.mark.asyncio
async def test_login_wrong_password_returns_none(auth_service, mock_user_repo):
    hashed_pw = bcrypt.hashpw("password".encode(), bcrypt.gensalt())
    mock_user_repo.get_by_username.return_value = {
        "user_id": 1,
        "username": "alice",
        "password_hash": hashed_pw.decode(),
    }

    token = await auth_service.login("alice", "wrongpassword")
    assert token is None


@pytest.mark.asyncio
async def test_login_nonexistent_user_returns_none(auth_service, mock_user_repo):
    mock_user_repo.get_by_username.return_value = None

    token = await auth_service.login("bob", "password")
    assert token is None


def test_create_access_token_contains_user_id(auth_service):
    token = auth_service._create_access_token(42)
    payload = jwt.decode(
        token, auth_service.jwt_secret, algorithms=[auth_service.jwt_algorithm]
    )
    assert payload["user_id"] == 42
    assert "exp" in payload
    # Ensure expiration is roughly correct
    exp = datetime.utcfromtimestamp(payload["exp"])
    now = datetime.utcnow()
    assert timedelta(minutes=59) < (exp - now) < timedelta(minutes=61)
