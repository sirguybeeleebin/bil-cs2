import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import StrEnum
from functools import wraps
from uuid import UUID, uuid4

import asyncpg
import jwt
import uvicorn
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette import status

load_dotenv()
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_WORKERS = int(os.getenv("APP_WORKERS", 1))
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info")

logging.basicConfig(level=APP_LOG_LEVEL.upper())
log = logging.getLogger("auth_service")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt_secret_key")
JWT_ALGORITHM = os.getenv("JWT_AL", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "cs2_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "cs2_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cs2_db")
POSTGRES_POOL_MIN_SIZE = int(os.getenv("POSTGRES_POOL_MIN_SIZE", 1))
POSTGRES_POOL_MAX_SIZE = int(os.getenv("POSTGRES_POOL_MAX_SIZE", 10))
POSTGRES_POOL_MAX_IDLE = int(os.getenv("POSTGRES_POOL_MAX_IDLE", 10))
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "auth")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

ph = PasswordHasher()


class Audience(StrEnum):
    AUTH = "AUTH"
    ADMIN = "ADMIN"
    USER = "USER"
    ETL_DICTIONARY = "ETL_DICTIONARY"
    DICTIONARY = "DICTIONARY"
    ETL_ML = "ETL_ML"
    ML = "ML"
    FORECAST = "FORECAST"


def authorize(expected_audience: Audience):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            auth_header: str | None = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Отсутствует токен авторизации",
                )
            token: str = auth_header.split(" ")[1]
            try:
                payload: dict = jwt.decode(
                    token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
                )
                user_audience = payload.get("aud", Audience.AUTH.value)
                if user_audience != expected_audience.value:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Нет доступа к этому ресурсу",
                    )
                request.state.user = payload
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Срок действия токена истёк",
                )
            except jwt.InvalidTokenError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Недействительный токен: {str(e)}",
                )
            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator


class RegisterRequest(BaseModel):
    username: str = Field(
        title="Имя пользователя", examples=["user123"], default="user123"
    )
    password: str = Field(title="Пароль", examples=["P@ssw0rd!"], default="P@ssw0rd!")
    role: Audience = Field(title="Роль пользователя", default=Audience.USER)


class RegisterResponse(BaseModel):
    user_id: UUID = Field(title="ID пользователя", examples=[uuid4()])


class LoginRequest(BaseModel):
    username: str = Field(
        title="Имя пользователя", examples=["user123"], default="user123"
    )
    password: str = Field(title="Пароль", examples=["P@ssw0rd!"], default="P@ssw0rd!")


class LoginResponse(BaseModel):
    access_token: str = Field(title="JWT токен доступа")
    token_type: str = Field(default="bearer")


class UserAlreadyExists(Exception):
    pass


class UserDoesNotExist(Exception):
    pass


class UserInvalidPassword(Exception):
    pass


class UserRepository:
    def __init__(self, pool: asyncpg.Pool, schema: str):
        self.pool = pool
        self.schema = schema

    async def get_by_username(self, username: str) -> dict | None:
        query = f"""
            SELECT user_id, username, password_hash, role, created_at, updated_at
            FROM {self.schema}.users
            WHERE username=$1
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, username)
                if row:
                    log.info("Пользователь '%s' найден в базе данных.", username)
                else:
                    log.info("Пользователь '%s' не найден в базе данных.", username)
                return dict(row) if row else None
        except Exception as e:
            log.error("Ошибка при получении пользователя '%s': %s", username, e)
            return None

    async def upsert_user(
        self, username: str, hashed_password: str, role: str = Audience.USER.value
    ) -> dict | None:
        query = f"""
            INSERT INTO {self.schema}.users(username, password_hash, role, created_at, updated_at)
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (username) DO UPDATE
            SET password_hash = EXCLUDED.password_hash,
                role = EXCLUDED.role,
                updated_at = NOW()
            RETURNING user_id, username, role
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, username, hashed_password, role)
                log.info(
                    "Пользователь '%s' успешно добавлен/обновлён в базе данных.",
                    username,
                )
                return dict(row) if row else None
        except Exception as e:
            log.error(
                "Ошибка при добавлении/обновлении пользователя '%s': %s", username, e
            )
            return None


class AuthService:
    def __init__(
        self,
        user_repository: UserRepository,
        jwt_secret_key: str,
        jwt_algorithm: str,
        jwt_access_token_expire_minutes: int,
    ):
        self.user_repository = user_repository
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
        self.jwt_access_token_expire_minutes = jwt_access_token_expire_minutes

    async def register(
        self, username: str, password: str, role: str = Audience.USER.value
    ) -> dict:
        existing_user = await self.user_repository.get_by_username(username)
        if existing_user:
            log.warning(
                "Попытка зарегистрировать уже существующего пользователя '%s'.",
                username,
            )
            raise UserAlreadyExists
        hashed_password = ph.hash(password)
        new_user = await self.user_repository.upsert_user(
            username, hashed_password, role
        )
        log.info("Пользователь '%s' успешно зарегистрирован.", username)
        return new_user

    async def login(self, username: str, password: str) -> str:
        db_user = await self.user_repository.get_by_username(username)
        if not db_user:
            log.warning("Попытка входа несуществующего пользователя '%s'.", username)
            raise UserDoesNotExist
        try:
            ph.verify(db_user["password_hash"], password)
        except VerifyMismatchError:
            log.warning("Неверный пароль для пользователя '%s'.", username)
            raise UserInvalidPassword
        expire = datetime.utcnow() + timedelta(
            minutes=self.jwt_access_token_expire_minutes
        )
        payload = {
            "user_id": str(db_user["user_id"]),
            "aud": db_user.get("role", Audience.USER.value),
            "exp": expire,
        }
        token = jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        log.info("Пользователь '%s' успешно вошёл в систему.", username)
        return token


router = APIRouter(prefix=API_PREFIX)


@router.post("/register", response_model=RegisterResponse)
async def register(user: RegisterRequest, request: Request):
    auth_service: AuthService = request.app.state.auth_service
    try:
        new_user = await auth_service.register(
            user.username, user.password, user.role.value
        )
        log.info("Регистрация пользователя '%s' прошла успешно.", user.username)
        return RegisterResponse(user_id=new_user["user_id"])
    except UserAlreadyExists:
        log.warning(
            "Попытка регистрации существующего пользователя '%s'.", user.username
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь уже существует",
        )


@router.post("/login", response_model=LoginResponse)
async def login(user: LoginRequest, request: Request):
    auth_service: AuthService = request.app.state.auth_service
    try:
        token = await auth_service.login(user.username, user.password)
        log.info("Вход пользователя '%s' выполнен успешно.", user.username)
        return LoginResponse(access_token=token)
    except UserDoesNotExist:
        log.warning("Попытка входа несуществующего пользователя '%s'.", user.username)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден"
        )
    except UserInvalidPassword:
        log.warning(
            "Попытка входа с неверным паролем для пользователя '%s'.", user.username
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный пароль"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "Подключение к базе данных PostgreSQL (%s:%s)...", POSTGRES_HOST, POSTGRES_PORT
    )
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=POSTGRES_POOL_MIN_SIZE,
        max_size=POSTGRES_POOL_MAX_SIZE,
        max_inactive_connection_lifetime=POSTGRES_POOL_MAX_IDLE,
    )
    log.info("Соединение с базой данных установлено.")
    app.state.user_repo = UserRepository(pool, schema=POSTGRES_SCHEMA)
    app.state.auth_service = AuthService(
        app.state.user_repo,
        JWT_SECRET_KEY,
        JWT_ALGORITHM,
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    )
    yield
    log.info("Закрытие пула соединений с базой данных...")
    await pool.close()
    log.info("Пул соединений закрыт.")


app = FastAPI(title="Auth Service", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        workers=APP_WORKERS,
        log_level=APP_LOG_LEVEL,
    )
