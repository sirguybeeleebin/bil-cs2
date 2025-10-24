from uuid import UUID

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from auth.services.service_auth import (
    InvalidServiceSecretError,
    ServiceAlreadyExistsError,
    ServiceAuthService,
    ServiceNotFoundError,
)


class ServiceRegisterRequest(BaseModel):
    client_id: str = Field(
        ...,
        title="ID сервиса",
        description="Уникальный идентификатор сервиса для регистрации",
        example="etl_001",
    )
    client_secret: str = Field(
        ...,
        title="Секрет сервиса",
        description="Секрет сервиса для аутентификации",
        example="StrongSecret123!",
    )


class ServiceLoginRequest(BaseModel):
    client_id: str = Field(
        ...,
        title="ID сервиса",
        description="Уникальный идентификатор сервиса для входа",
        example="etl_001",
    )
    client_secret: str = Field(
        ...,
        title="Секрет сервиса",
        description="Секрет сервиса для входа",
        example="StrongSecret123!",
    )


class ServiceLoginResponse(BaseModel):
    access_token: str = Field(
        ...,
        title="Токен доступа",
        description="JWT токен для авторизации сервиса",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    )
    token_type: str = Field(
        "bearer",
        title="Тип токена",
        description="Тип токена, обычно 'bearer'",
        example="bearer",
    )


def make_service_auth_router(service_auth_service: ServiceAuthService) -> APIRouter:
    router = APIRouter(prefix="/auth/service", tags=["AuthService"])

    @router.post("/register", response_model=ServiceRegisterRequest)
    async def register_service(data: ServiceRegisterRequest):
        try:
            return await service_auth_service.register_service(
                data.client_id, data.client_secret
            )
        except ServiceAlreadyExistsError:
            raise HTTPException(status_code=400, detail="Сервис уже существует")

    @router.post("/token", response_model=ServiceLoginResponse)
    async def service_token(data: ServiceLoginRequest):
        try:
            token = await service_auth_service.authenticate_service(
                data.client_id, data.client_secret
            )
            return {"access_token": token, "token_type": "bearer"}
        except (ServiceNotFoundError, InvalidServiceSecretError):
            raise HTTPException(status_code=401, detail="Неверные учетные данные")

    @router.get("/me")
    async def me(
        service_id: UUID = Header(
            ..., title="ID сервиса", description="ID текущего сервиса"
        ),
    ):
        try:
            return await service_auth_service.get_me(service_id)
        except ServiceNotFoundError:
            raise HTTPException(status_code=404, detail="Сервис не найден")

    return router
