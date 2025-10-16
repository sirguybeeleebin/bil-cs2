import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aio_pika
import uvicorn
from aio_pika import ExchangeType
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict


# ------------------ Settings ------------------
class Settings(BaseSettings):
    # ================= RabbitMQ =================
    RABBITMQ_USER: str = "cs2_user"
    RABBITMQ_PASSWORD: str = "cs2_password"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_QUEUE: str = "cs2_consumer"
    RABBITMQ_EXCHANGE: str = "cs2"
    RABBITMQ_EXCHANGE_TYPE: str = "direct"

    # Routing keys
    RABBITMQ_ROUTING_KEY_SPLIT: str = "cs2.split_created"
    RABBITMQ_ROUTING_KEY_ML: str = "cs2.ml_completed"

    # ================= App host/port =================
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_RELOAD: bool = True

    # ================= Logging =================
    LOG_LEVEL: str = "INFO"

    @property
    def RABBITMQ_URL(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{5672}/%2F"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# ------------------ Helper ------------------
def get_settings(env_file: str = ".env") -> Settings:
    env_path = Path(env_file)
    if env_path.exists():
        print(f"Loading configuration from {env_path}")
        return Settings(_env_file=env_path)
    print(f"Env file not found at {env_path}, using defaults")
    return Settings()


# ------------------ App Factory ------------------
def create_app(settings: Settings) -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ---------- STARTUP ----------
        connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        exchange = await channel.declare_exchange(
            settings.RABBITMQ_EXCHANGE, ExchangeType.DIRECT
        )
        queue = await channel.declare_queue(settings.RABBITMQ_QUEUE, durable=True)

        routing_keys = [
            settings.RABBITMQ_ROUTING_KEY_SPLIT,
            settings.RABBITMQ_ROUTING_KEY_ML,
        ]
        for rk in routing_keys:
            await queue.bind(exchange, routing_key=rk)

        async def event_handler(message: aio_pika.IncomingMessage):
            async with message.process():
                log.info(
                    f"[x] Received message [{message.routing_key}]: {message.body.decode()}"
                )

        consumer_task = asyncio.create_task(queue.consume(event_handler))
        log.info(f"[*] Consumer started for routing keys: {routing_keys}")

        yield

        # ---------- SHUTDOWN ----------
        if consumer_task:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        if connection:
            await connection.close()
        log.info("[*] RabbitMQ consumer stopped.")

    app = FastAPI(title="CS2 FastAPI RabbitMQ", lifespan=lifespan)
    return app


# ------------------ Uvicorn Runner ------------------
def run_uvicorn(app: FastAPI, settings: Settings):
    uvicorn.run(
        app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.APP_RELOAD,
    )


# ------------------ Main ------------------
def main():
    settings = get_settings()
    app = create_app(settings)
    run_uvicorn(app, settings)


if __name__ == "__main__":
    main()
