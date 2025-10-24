import logging

import asyncpg
import pytest
from repositories.service import make_service_repository
from testcontainers.postgres import PostgresContainer

logger = logging.getLogger("test_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


async def setup_database():
    postgres = PostgresContainer("postgres:15")
    postgres.start()
    dsn = postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")
    logger.info("PostgreSQL container started with DSN: %s", dsn)
    pool = await asyncpg.create_pool(dsn=dsn)
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE SCHEMA IF NOT EXISTS auth;
        CREATE TABLE IF NOT EXISTS auth.services (
            service_id SERIAL PRIMARY KEY,
            client_id TEXT UNIQUE NOT NULL,
            client_secret_hash TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """)
    service_repo = make_service_repository(pool)
    return postgres, pool, service_repo


@pytest.mark.asyncio
async def test_upsert_service():
    logger.info("Starting test_upsert_service")
    postgres, pool, service_repo = await setup_database()
    logger.info("Upserting service 'client1' with secret 'secret123'")
    service = await service_repo.upsert_service("client1", "secret123")
    assert service is not None
    logger.info("Service inserted: %s", service)
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_upsert_service")


@pytest.mark.asyncio
async def test_get_service():
    logger.info("Starting test_get_service")
    postgres, pool, service_repo = await setup_database()
    await service_repo.upsert_service("client1", "secret123")
    fetched_service = await service_repo.get_service_by_client_id("client1")
    assert fetched_service is not None
    assert fetched_service["client_id"] == "client1"
    logger.info("Service fetched: %s", fetched_service)
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_get_service")


@pytest.mark.asyncio
async def test_update_service_secret():
    logger.info("Starting test_update_service_secret")
    postgres, pool, service_repo = await setup_database()
    service = await service_repo.upsert_service("client1", "secret123")
    updated_service = await service_repo.upsert_service("client1", "newsecret456")
    assert updated_service["service_id"] == service["service_id"]
    logger.info("Service updated: %s", updated_service)
    fetched_service = await service_repo.get_service_by_client_id("client1")
    assert fetched_service["client_secret_hash"] == "newsecret456"
    logger.info("Secret verified: %s", fetched_service["client_secret_hash"])
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_update_service_secret")


@pytest.mark.asyncio
async def test_get_nonexistent_service():
    logger.info("Starting test_get_nonexistent_service")
    postgres, pool, service_repo = await setup_database()
    service = await service_repo.get_service_by_client_id("nonexistent")
    assert service is None
    logger.info("Nonexistent service correctly returned None")
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_get_nonexistent_service")
