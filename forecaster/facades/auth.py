# auth/facade.py
import httpx
from fastapi import HTTPException
from starlette import status


class AuthFacade:
    """
    Фасад для взаимодействия с Auth микросервисом
    """

    def __init__(self, auth_service_url: str):
        self.auth_service_url = auth_service_url.rstrip("/")

    async def verify_token(self, token: str) -> dict:
        """
        Проверяет JWT токен через Auth сервис и возвращает данные пользователя
        """
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self.auth_service_url}/auth/verify",
                    json={"token": token},
                    timeout=5.0,
                )
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Auth service unavailable: {e}",
                )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )

        return resp.json()  # {"user_id": ..., "username": ...}
