import re
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field, validator

from auth.services.auth import (
    AuthService,
    InvalidServiceSecretError,
    InvalidUserPasswordError,
    ServiceAlreadyExistsError,
    ServiceNotFoundError,
    UserAlreadyExistsError,
    UserNotFoundError,
)


class UserRegisterRequest(BaseModel):
    username: str = Field(
        ...,
        example="user1",
        title="Имя пользователя",
        description="Уникальное имя пользователя",
    )
    password: str = Field(
        ...,
        example="StrongPassword123",
        title="Пароль",
        description="Пароль пользователя",
    )

    @validator("username")
    def validate_username(cls, v):
        if not re.fullmatch(r"[a-zA-Z0-9_]{3,30}", v):
            raise ValueError(
                "Имя пользователя должно быть 3-30 символов, только латиница, цифры и _"
            )
        return v

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Пароль должен быть не менее 8 символов")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Пароль должен содержать хотя бы одну заглавную букву")
        if not re.search(r"[a-z]", v):
            raise ValueError("Пароль должен содержать хотя бы одну строчную букву")
        if not re.search(r"\d", v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        return v


class UserRegisterResponse(BaseModel):
    user_id: UUID = Field(
        ...,
        title="ID пользователя",
        description="Уникальный идентификатор пользователя",
    )
    username: str = Field(
        ...,
        title="Имя пользователя",
        description="Имя пользователя",
    )
    created_at: datetime = Field(
        ...,
        title="Дата создания",
        description="Дата и время создания пользователя",
    )
    updated_at: datetime = Field(
        ...,
        title="Дата обновления",
        description="Дата и время последнего обновления пользователя",
    )


class UserLoginRequest(BaseModel):
    username: str = Field(
        ...,
        title="Имя пользователя",
        description="Имя пользователя",
    )
    password: str = Field(
        ...,
        title="Пароль",
        description="Пароль пользователя",
    )


class UserLoginResponse(BaseModel):
    access_token: str = Field(
        ...,
        title="Токен доступа",
        description="JWT токен для авторизации",
    )
    token_type: str = Field(
        "bearer",
        title="Тип токена",
        description="Тип токена, обычно 'bearer'",
    )


class ServiceRegisterRequest(BaseModel):
    client_id: str = Field(
        ...,
        example="etl_001",
        title="ID сервиса",
        description="Уникальный идентификатор сервиса",
    )
    client_secret: str = Field(
        ...,
        example="some-secure-secret",
        title="Секрет сервиса",
        description="Секрет сервиса для аутентификации",
    )


class ServiceRegisterResponse(BaseModel):
    service_id: UUID = Field(
        ...,
        title="ID сервиса",
        description="Уникальный идентификатор сервиса",
    )
    client_id: str = Field(
        ...,
        title="ID сервиса",
        description="Идентификатор сервиса",
    )
    created_at: datetime = Field(
        ...,
        title="Дата создания",
        description="Дата и время создания сервиса",
    )
    updated_at: datetime = Field(
        ...,
        title="Дата обновления",
        description="Дата и время последнего обновления сервиса",
    )


class ServiceLoginRequest(BaseModel):
    client_id: str = Field(
        ...,
        title="ID сервиса",
        description="Идентификатор сервиса",
    )
    client_secret: str = Field(
        ...,
        title="Секрет сервиса",
        description="Секрет сервиса для аутентификации",
    )


class ServiceLoginResponse(BaseModel):
    access_token: str = Field(
        ...,
        title="Токен доступа",
        description="JWT токен для сервиса",
    )
    token_type: str = Field(
        "bearer",
        title="Тип токена",
        description="Тип токена, обычно 'bearer'",
    )


def make_auth_router(auth_service: AuthService) -> APIRouter:
    router = APIRouter(prefix="/auth")

    @router.post("/register", response_model=UserRegisterResponse)
    async def register(data: UserRegisterRequest):
        try:
            return await auth_service.register_user(data.username, data.password)
        except UserAlreadyExistsError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Пользователь уже существует",
            )

    @router.post("/login", response_model=UserLoginResponse)
    async def login(data: UserLoginRequest):
        try:
            token = await auth_service.authenticate_user(data.username, data.password)
            return {"access_token": token, "token_type": "bearer"}
        except (UserNotFoundError, InvalidUserPasswordError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные",
            )

    @router.post("/service/register", response_model=ServiceRegisterResponse)
    async def register_service(data: ServiceRegisterRequest):
        try:
            return await auth_service.register_service(
                data.client_id, data.client_secret
            )
        except ServiceAlreadyExistsError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Сервис уже существует",
            )

    @router.post("/service/token", response_model=ServiceLoginResponse)
    async def service_token(data: ServiceLoginRequest):
        try:
            token = await auth_service.authenticate_service(
                data.client_id, data.client_secret
            )
            return {"access_token": token, "token_type": "bearer"}
        except (ServiceNotFoundError, InvalidServiceSecretError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные",
            )

    @router.get("/me", response_model=UserRegisterResponse)
    async def get_me(
        user_id: UUID = Header(
            ..., title="ID пользователя", description="ID текущего пользователя"
        ),
    ):
        try:
            return await auth_service.get_me(user_id)
        except UserNotFoundError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    return router
