import re
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, validator

from auth.services.auth import (
    AuthService,
    InvalidUserPasswordError,
    UserAlreadyExistsError,
    UserNotFoundError,
)


class UserRegisterRequest(BaseModel):
    username: str = Field(
        ...,
        title="Имя пользователя",
        description="Уникальное имя пользователя, 3-30 символов, только латиница, цифры и _",
        example="user_123",
    )
    password: str = Field(
        ...,
        title="Пароль",
        description="Пароль пользователя, минимум 8 символов, с заглавной буквой, строчной и цифрой",
        example="StrongPass1",
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
        if (
            len(v) < 8
            or not re.search(r"[A-Z]", v)
            or not re.search(r"[a-z]", v)
            or not re.search(r"\d", v)
        ):
            raise ValueError(
                "Пароль должен быть не менее 8 символов, содержать хотя бы одну заглавную букву, одну строчную и одну цифру"
            )
        return v


class UserLoginRequest(BaseModel):
    username: str = Field(
        ...,
        title="Имя пользователя",
        description="Имя пользователя для входа",
        example="user_123",
    )
    password: str = Field(
        ..., title="Пароль", description="Пароль пользователя", example="StrongPass1"
    )


class UserLoginResponse(BaseModel):
    access_token: str = Field(
        ...,
        title="Токен доступа",
        description="JWT токен для авторизации",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    )
    token_type: str = Field(
        "bearer",
        title="Тип токена",
        description="Тип токена, обычно 'bearer'",
        example="bearer",
    )


def make_auth_router(auth_service: AuthService) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["Auth"])

    @router.post("/register")
    async def register(data: UserRegisterRequest):
        try:
            return await auth_service.register_user(data.username, data.password)
        except UserAlreadyExistsError:
            raise HTTPException(status_code=400, detail="Пользователь уже существует")

    @router.post("/login", response_model=UserLoginResponse)
    async def login(data: UserLoginRequest):
        try:
            token = await auth_service.authenticate_user(data.username, data.password)
            return {"access_token": token}
        except (UserNotFoundError, InvalidUserPasswordError):
            raise HTTPException(status_code=401, detail="Неверные учетные данные")

    @router.get("/me")
    async def me(user_id: UUID = Header(...)):
        try:
            return await auth_service.get_me(user_id)
        except UserNotFoundError:
            raise HTTPException(status_code=404, detail="Пользователь не найден")

    return router
