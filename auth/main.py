from fastapi import FastAPI, APIRouter
import asyncpg
from pydantic_settings import BaseSettings, SettingsConfigDict
import uvicorn

from routers.auth import make_auth_router
from repositories.user import make_user_repository
from services.auth import make_auth_service


class Settings(BaseSettings):
    TITLE: str = "Auth Service"
    VERSION: str = "/api/v1"
    
    JWT_SECRET: str = "supersecret"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    
    POSTGRES_DSN: str
    POSTGRES_POOL_MIN_SIZE: int = 1
    POSTGRES_POOL_MAX_SIZE: int = 10
    POSTGRES_POOL_MAX_IDLE: float = 60.0  

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()


async def lifespan(app: FastAPI):    
    pool = await asyncpg.create_pool(
        dsn=settings.POSTGRES_DSN,
        min_size=settings.POSTGRES_POOL_MIN_SIZE,
        max_size=settings.POSTGRES_POOL_MAX_SIZE,
        max_inactive_connection_lifetime=settings.POSTGRES_POOL_MAX_IDLE,
    )

    user_repo = make_user_repository(pool)
    
    auth_service = make_auth_service(
        user_repository=user_repo,
        jwt_secret=settings.JWT_SECRET,
        jwt_algorithm=settings.JWT_ALGORITHM,
        access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )

    auth_router = make_auth_router(auth_service)
    
    router = APIRouter(prefix=settings.VERSION)
    router.include_router(auth_router)    
    
    app.include_router(router)    

    yield

    await pool.close()


app = FastAPI(title=settings.TITLE, lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
