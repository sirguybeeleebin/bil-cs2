import logging

import asyncpg

log = logging.getLogger("user_repository")


class UserRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_by_username(self, username: str):
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, username, password_hash FROM users WHERE username = $1",
                    username,
                )
                return dict(row) if row else None
        except asyncpg.PostgresError as e:
            log.error(f"Database error in get_by_username: {e}")
            return None
        except Exception as e:
            log.exception(f"Unexpected error in get_by_username: {e}")
            return None

    async def upsert(self, username: str, password_hash: str):
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO users(username, password_hash)
                    VALUES ($1, $2)
                    ON CONFLICT(username)
                    DO UPDATE SET password_hash = EXCLUDED.password_hash
                    RETURNING id, username
                    """,
                    username,
                    password_hash,
                )
                return dict(row)
        except asyncpg.PostgresError as e:
            log.error(f"Database error in upsert: {e}")
            return None
        except Exception as e:
            log.exception(f"Unexpected error in upsert: {e}")
            return None


def make_user_repository(pool: asyncpg.Pool) -> UserRepository:
    return UserRepository(pool)
