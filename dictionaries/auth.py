from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Header, HTTPException, status

from dictionaries.config import AUTH_SERVICE_URL


async def lifespan(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        app.state.http_client = client
        yield


async def validate_token(
    authorization: str | None = Header(None), app: FastAPI | None = None
) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Отсутствует или неверный заголовок Bearer",
        )

    token = authorization.split(" ", 1)[1]
    client: httpx.AsyncClient = app.state.http_client
    resp = await client.get(
        AUTH_SERVICE_URL, headers={"Authorization": f"Bearer {token}"}
    )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный или просроченный токен",
        )
