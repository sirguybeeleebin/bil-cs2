import asyncpg


class UserRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def upsert_user(self, username: str, password_hash: str) -> dict | None:
        query = """
        INSERT INTO auth.users (username, password_hash, created_at, updated_at)
        VALUES ($1, $2, NOW(), NOW())
        ON CONFLICT (username) DO UPDATE
        SET password_hash = EXCLUDED.password_hash,
            updated_at = NOW()
        RETURNING user_id, username, created_at, updated_at
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, username, password_hash)
            return dict(row)
        except asyncpg.PostgresError:
            return None

    async def get_user_by_username(self, username: str) -> dict | None:
        query = """
        SELECT user_id, username, password_hash, created_at, updated_at
        FROM auth.users
        WHERE username = $1
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, username)
            return dict(row) if row else None
        except asyncpg.PostgresError:
            return None


def make_user_repository(pool: asyncpg.Pool) -> UserRepository:
    return UserRepository(pool)
