from unittest.mock import AsyncMock
from uuid import uuid4

import bcrypt
import pytest

from auth.services.service_auth import (
    InvalidServiceSecretError,
    ServiceAlreadyExistsError,
    ServiceAuthService,
)

JWT_SECRET = "secret"
JWT_ALGO = "HS256"
TOKEN_EXPIRE = 60


@pytest.mark.asyncio
async def test_register_service_success():
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = None
    service_repo.upsert_service.return_value = {
        "service_id": uuid4(),
        "client_id": "etl_001",
    }

    service_auth = ServiceAuthService(service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    result = await service_auth.register_service("etl_001", "Secret123!")

    assert result is not None
    assert result["client_id"] == "etl_001"
    service_repo.get_service_by_client_id.assert_awaited_once_with("etl_001")
    service_repo.upsert_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_service_existing():
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = {
        "service_id": uuid4(),
        "client_id": "etl_001",
    }

    service_auth = ServiceAuthService(service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(ServiceAlreadyExistsError):
        await service_auth.register_service("etl_001", "Secret123!")


@pytest.mark.asyncio
async def test_authenticate_service_wrong_secret():
    secret = "Secret123!"
    hashed_secret = bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = {
        "service_id": uuid4(),
        "client_secret_hash": hashed_secret,
    }

    service_auth = ServiceAuthService(service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    with pytest.raises(InvalidServiceSecretError):
        await service_auth.authenticate_service("etl_001", "WrongSecret")


@pytest.mark.asyncio
async def test_authenticate_service_success():
    secret = "Secret123!"
    hashed_secret = bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()
    service_repo = AsyncMock()
    service_repo.get_service_by_client_id.return_value = {
        "service_id": uuid4(),
        "client_secret_hash": hashed_secret,
    }

    service_auth = ServiceAuthService(service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    token = await service_auth.authenticate_service("etl_001", secret)

    assert token is not None
    service_repo.get_service_by_client_id.assert_awaited_once_with("etl_001")


@pytest.mark.asyncio
async def test_get_me_success():
    service_id = uuid4()
    service_repo = AsyncMock()
    service_repo.get_service_by_id.return_value = {"service_id": service_id}

    service_auth = ServiceAuthService(service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)
    result = await service_auth.get_me(service_id)

    assert result["service_id"] == service_id
    service_repo.get_service_by_id.assert_awaited_once_with(service_id)


@pytest.mark.asyncio
async def test_get_me_not_found():
    service_repo = AsyncMock()
    service_repo.get_service_by_id.return_value = None

    service_auth = ServiceAuthService(service_repo, JWT_SECRET, JWT_ALGO, TOKEN_EXPIRE)

    from auth.services.service_auth import ServiceNotFoundError

    with pytest.raises(ServiceNotFoundError):
        await service_auth.get_me(uuid4())
