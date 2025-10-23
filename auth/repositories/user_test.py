import logging

import asyncpg
import pytest
from testcontainers.postgres import PostgresContainer

from auth.repositories.user import make_user_repository

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
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """)
    user_repo = make_user_repository(pool)
    return postgres, pool, user_repo


@pytest.mark.asyncio
async def test_upsert_user():
    logger.info("Starting test_upsert_user")
    postgres, pool, user_repo = await setup_database()
    logger.info("Upserting user 'testuser' with password 'password123'")
    user = await user_repo.upsert_user("testuser", "password123")
    assert user is not None
    logger.info("User inserted: %s", user)
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_upsert_user")


@pytest.mark.asyncio
async def test_get_user():
    logger.info("Starting test_get_user")
    postgres, pool, user_repo = await setup_database()
    await user_repo.upsert_user("testuser", "password123")
    fetched_user = await user_repo.get_user_by_username("testuser")
    assert fetched_user is not None
    assert fetched_user["username"] == "testuser"
    logger.info("User fetched: %s", fetched_user)
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_get_user")


@pytest.mark.asyncio
async def test_update_user_password():
    logger.info("Starting test_update_user_password")
    postgres, pool, user_repo = await setup_database()
    user = await user_repo.upsert_user("testuser", "password123")
    updated_user = await user_repo.upsert_user("testuser", "newpass456")
    assert updated_user["id"] == user["id"]
    logger.info("User updated: %s", updated_user)
    fetched_user = await user_repo.get_user_by_username("testuser")
    assert fetched_user["password"] == "newpass456"
    logger.info("Password verified: %s", fetched_user["password"])
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_update_user_password")


@pytest.mark.asyncio
async def test_get_nonexistent_user():
    logger.info("Starting test_get_nonexistent_user")
    postgres, pool, user_repo = await setup_database()
    user = await user_repo.get_user_by_username("nonexistent")
    assert user is None
    logger.info("Nonexistent user correctly returned None")
    await pool.close()
    postgres.stop()
    logger.info("Test finished: test_get_nonexistent_user")
