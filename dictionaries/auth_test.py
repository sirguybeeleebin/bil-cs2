from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException

from dictionaries.auth import lifespan, validate_token


@asynccontextmanager
async def lifespan_cm(app: FastAPI):
    async for _ in lifespan(app):
        yield


@pytest.mark.asyncio
async def test_validate_token_success():
    app = FastAPI()

    # Mock the client.get call to return a successful response
    async with lifespan_cm(app):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.status_code = 200
            await validate_token(authorization="Bearer validtoken", app=app)


@pytest.mark.asyncio
async def test_validate_token_missing_header():
    app = FastAPI()
    async with lifespan_cm(app):
        with pytest.raises(HTTPException) as exc_info:
            await validate_token(authorization=None, app=app)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Отсутствует или неверный заголовок Bearer"


@pytest.mark.asyncio
async def test_validate_token_invalid_token():
    app = FastAPI()
    async with lifespan_cm(app):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.status_code = 401
            with pytest.raises(HTTPException) as exc_info:
                await validate_token(authorization="Bearer invalidtoken", app=app)
            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Неверный или просроченный токен"
