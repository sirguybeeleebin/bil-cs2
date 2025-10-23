import asyncio
import logging

import asyncpg
import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from auth.repositories.user import UserRepository  

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------
# Fix for pytest-asyncio scope mismatch
# ------------------------
@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# ------------------------
# PostgreSQL container fixture
# ------------------------
@pytest_asyncio.fixture(scope="module")
async def postgres_container():
    logger.info("Starting PostgreSQL container...")
    container = PostgresContainer("postgres:15")
    container.start()
    dsn = container.get_connection_url().replace("+psycopg2", "")
    logger.info(f"PostgreSQL container DSN: {dsn}")

    pool = None
    for i in range(30):
        try:
            pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1;")
            logger.info("PostgreSQL is ready!")
            break
        except Exception as e:
            logger.debug(f"Attempt {i+1}: PostgreSQL not ready yet ({e})")
            await asyncio.sleep(1)
    else:
        raise RuntimeError("PostgreSQL container did not start in time")

    # Ensure users table exists
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(256) NOT NULL
        );
        """)
    yield pool

    logger.info("Closing PostgreSQL pool...")
    await pool.close()
    logger.info("Stopping PostgreSQL container...")
    container.stop()

# ------------------------
# UserRepository fixture
# ------------------------
@pytest_asyncio.fixture
async def user_repo(postgres_container: asyncpg.Pool):
    repo = UserRepository(pool=postgres_container)
    yield repo
    # Clean up table after each test
    async with postgres_container.acquire() as conn:
        await conn.execute("TRUNCATE TABLE users;")
        logger.info("'users' table cleaned after test.")

# ------------------------
# Tests
# ------------------------
@pytest.mark.asyncio
async def test_upsert_and_get_by_username(user_repo: UserRepository):
    logger.info("Running test_upsert_and_get_by_username...")

    username = "testuser"
    password_hash = "hashedpassword123"

    # Test upsert
    result = await user_repo.upsert(username, password_hash)
    assert result is not None
    assert result["username"] == username
    assert "id" in result
    logger.info("Upsert successful.")

    # Test get_by_username
    user = await user_repo.get_by_username(username)
    assert user is not None
    assert user["username"] == username
    assert "password_hash" in user
    logger.info("get_by_username returned expected result.")

    # Test get_by_username for non-existent user
    missing = await user_repo.get_by_username("nonexistent")
    assert missing is None
    logger.info("get_by_username correctly returned None for missing user.")

@pytest.mark.asyncio
async def test_upsert_updates_password(user_repo: UserRepository):
    username = "updateuser"
    password1 = "firsthash"
    password2 = "secondhash"

    # Insert first time
    res1 = await user_repo.upsert(username, password1)
    assert res1 is not None

    # Update password
    res2 = await user_repo.upsert(username, password2)
    assert res2 is not None
    assert res2["username"] == username

    # Fetch and verify password_hash updated
    user = await user_repo.get_by_username(username)
    assert user["password_hash"] == password2
    logger.info("Password hash updated correctly on upsert.")
