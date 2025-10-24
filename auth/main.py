import os

import asyncpg
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from auth.middlewares.logging import LoggingMiddleware
from auth.repositories.service import make_service_repository
from auth.repositories.user import make_user_repository
from auth.routers.auth import make_auth_router
from auth.services.auth import make_auth_service

load_dotenv()

# Config
TITLE = os.environ.get("AUTH_APP_TITLE", "Auth Service")
VERSION = os.environ.get("AUTH_APP_VERSION", "/api/v1")
JWT_SECRET = os.environ.get("JWT_SECRET", "supersecret")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", 60))
APP_HOST = os.environ.get("AUTH_APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("AUTH_APP_PORT", 8000))

# Postgres config
POSTGRES_USER = os.environ.get("POSTGRES_USER", "cs2_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "cs2_password")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "cs2_db")
POSTGRES_HOST = os.environ.get("DOCKER_POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", 5432)
POSTGRES_POOL_MIN_SIZE = int(os.environ.get("POSTGRES_POOL_MIN_SIZE", 1))
POSTGRES_POOL_MAX_SIZE = int(os.environ.get("POSTGRES_POOL_MAX_SIZE", 10))
POSTGRES_POOL_MAX_IDLE = float(os.environ.get("POSTGRES_POOL_MAX_IDLE", 60.0))

# Form DSN
POSTGRES_DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(
        dsn=POSTGRES_DSN,
        min_size=POSTGRES_POOL_MIN_SIZE,
        max_size=POSTGRES_POOL_MAX_SIZE,
        max_inactive_connection_lifetime=POSTGRES_POOL_MAX_IDLE,
    )
    app.state.db_pool = pool

    user_repo = make_user_repository(pool)
    service_repo = make_service_repository(pool)
    auth_service = make_auth_service(
        user_repository=user_repo,
        service_repository=service_repo,
        jwt_secret=JWT_SECRET,
        jwt_algorithm=JWT_ALGORITHM,
        token_expire_minutes=JWT_EXPIRE_MINUTES,
    )

    auth_router = make_auth_router(auth_service)
    app.include_router(auth_router, prefix=VERSION)

    try:
        yield
    finally:
        await pool.close()


app = FastAPI(title=TITLE, lifespan=lifespan)
app.add_middleware(LoggingMiddleware)


@app.get("/health")
async def health():
    try:
        async with app.state.db_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        return {"status": "ok"}
    except Exception:
        return {"status": "fail"}


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=False)
