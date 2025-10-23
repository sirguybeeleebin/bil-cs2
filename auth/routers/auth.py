import re
import string

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from services.auth import AuthService


class RegisterRequest(BaseModel):
    username: str = Field(..., example="user_123!")
    password: str = Field(..., example="Secure123!")

    @validator("username")
    def username_valid(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError("Имя пользователя должно быть от 3 до 50 символов")
        allowed_chars = string.ascii_letters + string.digits + string.punctuation
        if any(c not in allowed_chars for c in v):
            raise ValueError(
                "Имя пользователя может содержать только английские буквы, цифры и символы"
            )
        return v

    @validator("password")
    def password_valid(cls, v):
        if len(v) < 8:
            raise ValueError("Пароль должен содержать не менее 8 символов")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Пароль должен содержать хотя бы одну заглавную букву")
        if not re.search(r"[a-z]", v):
            raise ValueError("Пароль должен содержать хотя бы одну строчную букву")
        if not re.search(r"\d", v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        if not re.search(r"[^\w\s]", v):
            raise ValueError("Пароль должен содержать хотя бы один специальный символ")
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., example="user_123!")
    password: str = Field(..., example="Secure123!")


class AuthResponse(BaseModel):
    token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")


def make_auth_router(auth_service: AuthService) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/register", response_model=AuthResponse)
    async def register(req: RegisterRequest):
        try:
            token = await auth_service.register(req.username, req.password)
            return AuthResponse(token=token)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/login", response_model=AuthResponse)
    async def login(req: LoginRequest):
        token = await auth_service.login(req.username, req.password)
        if not token:
            raise HTTPException(
                status_code=401, detail="Неверное имя пользователя или пароль"
            )
        return AuthResponse(token=token)

    return router
