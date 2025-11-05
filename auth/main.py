import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import asyncpg
import jwt
import uvicorn
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

load_dotenv()
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt_secret_key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "auth_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "auth_pass")
POSTGRES_DB = os.getenv("POSTGRES_DB", "auth_db")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "auth")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

logging.basicConfig(level=APP_LOG_LEVEL.upper())
log = logging.getLogger("auth_service")
ph = PasswordHasher()


class RegisterUserRequest(BaseModel):
    username: str = Field(
        ...,
        title="Имя пользователя",
        description="Уникальное имя пользователя для регистрации",
        examples=["user123", "ivan_petrov", "alice"],
    )
    password: str = Field(
        ...,
        title="Пароль",
        description="Пароль пользователя",
        examples=["Password123!", "qwerty2025", "MySecretPass"],
    )


class RegisterUserResponse(BaseModel):
    user_id: UUID = Field(
        ...,
        title="ID пользователя",
        description="Уникальный идентификатор зарегистрированного пользователя",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class LoginUserRequest(BaseModel):
    username: str = Field(
        ...,
        title="Имя пользователя",
        description="Имя пользователя для входа",
        examples=["user123", "ivan_petrov", "alice"],
    )
    password: str = Field(
        ...,
        title="Пароль",
        description="Пароль пользователя",
        examples=["Password123!", "qwerty2025", "MySecretPass"],
    )


class LoginUserResponse(BaseModel):
    access_token: str = Field(
        ...,
        title="Токен доступа",
        description="JWT токен для аутентификации",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )
    token_type: str = Field(
        "bearer",
        title="Тип токена",
        description="Тип токена (по умолчанию 'bearer')",
        examples=["bearer"],
    )


class RegisterServiceRequest(BaseModel):
    client_id: str = Field(
        ...,
        title="ID клиента",
        description="Уникальный идентификатор сервиса",
        examples=["service_123", "payment_service", "analytics"],
    )
    client_secret: str = Field(
        ...,
        title="Секрет клиента",
        description="Секрет для аутентификации сервиса",
        examples=["supersecret123", "payment2025", "serviceKey!"],
    )


class RegisterServiceResponse(BaseModel):
    service_id: UUID = Field(
        ...,
        title="ID сервиса",
        description="Уникальный идентификатор зарегистрированного сервиса",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


class LoginServiceRequest(BaseModel):
    client_id: str = Field(
        ...,
        title="ID клиента",
        description="Уникальный идентификатор сервиса для входа",
        examples=["service_123", "payment_service", "analytics"],
    )
    client_secret: str = Field(
        ...,
        title="Секрет клиента",
        description="Секрет для аутентификации сервиса",
        examples=["supersecret123", "payment2025", "serviceKey!"],
    )


class LoginServiceResponse(BaseModel):
    access_token: str = Field(
        ...,
        title="Токен доступа",
        description="JWT токен для аутентификации сервиса",
        examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
    )
    token_type: str = Field(
        "bearer",
        title="Тип токена",
        description="Тип токена (по умолчанию 'bearer')",
        examples=["bearer"],
    )


class UserRepository:
    def __init__(self, pool: asyncpg.Pool, schema: str):
        self.pool = pool
        self.schema = schema

    async def get_by_username(self, username: str) -> dict | None:
        query = f"SELECT user_id, username, password_hash FROM {self.schema}.users WHERE username=$1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, username)
            return dict(row) if row else None

    async def upsert_user(self, username: str, password_hash: str) -> dict:
        query = f"""
            INSERT INTO {self.schema}.users(user_id, username, password_hash, created_at, updated_at)
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (username) DO UPDATE
            SET password_hash = EXCLUDED.password_hash,
                updated_at = NOW()
            RETURNING user_id, username
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, uuid4(), username, password_hash)
            return dict(row)


class ServiceRepository:
    def __init__(self, pool: asyncpg.Pool, schema: str):
        self.pool = pool
        self.schema = schema

    async def get_by_client_id(self, client_id: str) -> dict | None:
        query = f"SELECT service_id, client_id, client_secret FROM {self.schema}.services WHERE client_id=$1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, client_id)
            return dict(row) if row else None

    async def upsert_service(self, client_id: str, client_secret: str) -> dict:
        query = f"""
            INSERT INTO {self.schema}.services(service_id, client_id, client_secret, created_at, updated_at)
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (client_id) DO UPDATE
            SET client_secret = EXCLUDED.client_secret,
                updated_at = NOW()
            RETURNING service_id, client_id
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, uuid4(), client_id, client_secret)
            return dict(row)


class UserExistsError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


class InvalidUserPasswordError(Exception):
    pass


class ServiceExistsError(Exception):
    pass


class ServiceNotFoundError(Exception):
    pass


class InvalidServiceSecretError(Exception):
    pass


class AuthService:
    def __init__(
        self,
        user_repo: UserRepository,
        service_repo: ServiceRepository,
        jwt_secret_key: str,
        jwt_algorithm: str,
        jwt_exp_minutes: int,
    ):
        self.user_repo = user_repo
        self.service_repo = service_repo
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
        self.jwt_exp_minutes = jwt_exp_minutes

    async def register_user(self, username: str, password: str) -> dict:
        existing = await self.user_repo.get_by_username(username)
        if existing:
            raise UserExistsError()
        hashed = ph.hash(password)
        user = await self.user_repo.upsert_user(username, hashed)
        return {"user_id": str(user["user_id"])}

    async def login_user(self, username: str, password: str) -> dict:
        user = await self.user_repo.get_by_username(username)
        if not user:
            raise UserNotFoundError()
        try:
            ph.verify(user["password_hash"], password)
        except VerifyMismatchError:
            raise InvalidUserPasswordError()
        exp = datetime.utcnow() + timedelta(minutes=self.jwt_exp_minutes)
        payload = {"user_id": str(user["user_id"]), "exp": exp, "type": "user"}
        token = jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        return {"access_token": token, "token_type": "bearer"}

    async def register_service(self, client_id: str, client_secret: str) -> dict:
        existing = await self.service_repo.get_by_client_id(client_id)
        if existing:
            raise ServiceExistsError()
        service = await self.service_repo.upsert_service(client_id, client_secret)
        return {"service_id": str(service["service_id"])}

    async def login_service(self, client_id: str, client_secret: str) -> dict:
        service = await self.service_repo.get_by_client_id(client_id)
        if not service:
            raise ServiceNotFoundError()
        if service["client_secret"] != client_secret:
            raise InvalidServiceSecretError()
        exp = datetime.utcnow() + timedelta(minutes=self.jwt_exp_minutes)
        payload = {
            "service_id": str(service["service_id"]),
            "exp": exp,
            "type": "service",
        }
        token = jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        return {"access_token": token, "token_type": "bearer"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(DATABASE_URL)
    app.state.user_repo = UserRepository(app.state.pool, schema=POSTGRES_SCHEMA)
    app.state.service_repo = ServiceRepository(app.state.pool, schema=POSTGRES_SCHEMA)
    app.state.auth_service = AuthService(
        app.state.user_repo,
        app.state.service_repo,
        JWT_SECRET_KEY,
        JWT_ALGORITHM,
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    yield
    await app.state.pool.close()


app = FastAPI(title="Auth Service", lifespan=lifespan)
router = APIRouter(prefix="/api/v1")


@router.post("/users/register", response_model=RegisterUserResponse)
async def register_user(request: RegisterUserRequest, http_request: Request):
    service: AuthService = http_request.app.state.auth_service
    try:
        data = await service.register_user(request.username, request.password)
        return RegisterUserResponse(**data)
    except UserExistsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь уже существует",
        )


@router.post("/users/login", response_model=LoginUserResponse)
async def login_user(request: LoginUserRequest, http_request: Request):
    service: AuthService = http_request.app.state.auth_service
    try:
        data = await service.login_user(request.username, request.password)
        return LoginUserResponse(**data)
    except UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден"
        )
    except InvalidUserPasswordError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный пароль"
        )


@router.post("/services/register", response_model=RegisterServiceResponse)
async def register_service(request: RegisterServiceRequest, http_request: Request):
    service: AuthService = http_request.app.state.auth_service
    try:
        data = await service.register_service(request.client_id, request.client_secret)
        return RegisterServiceResponse(**data)
    except ServiceExistsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Сервис уже существует"
        )


@router.post("/services/login", response_model=LoginServiceResponse)
async def login_service(request: LoginServiceRequest, http_request: Request):
    service: AuthService = http_request.app.state.auth_service
    try:
        data = await service.login_service(request.client_id, request.client_secret)
        return LoginServiceResponse(**data)
    except ServiceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Сервис не найден"
        )
    except InvalidServiceSecretError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный секрет сервиса"
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, log_level=APP_LOG_LEVEL)
