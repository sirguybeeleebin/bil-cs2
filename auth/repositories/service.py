import asyncpg


class ServiceRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def upsert_service(
        self, client_id: str, client_secret_hash: str
    ) -> dict | None:
        query = """
        INSERT INTO auth.services (client_id, client_secret_hash, created_at, updated_at)
        VALUES ($1, $2, NOW(), NOW())
        ON CONFLICT (client_id) DO UPDATE
        SET client_secret_hash = EXCLUDED.client_secret_hash,
            updated_at = NOW()
        RETURNING service_id, client_id, created_at, updated_at
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, client_id, client_secret_hash)
            return dict(row)
        except asyncpg.PostgresError:
            return None

    async def get_service_by_client_id(self, client_id: str) -> dict | None:
        query = """
        SELECT service_id, client_id, client_secret_hash, created_at, updated_at
        FROM auth.services
        WHERE client_id = $1
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, client_id)
            return dict(row) if row else None
        except asyncpg.PostgresError:
            return None


def make_service_repository(pool: asyncpg.Pool) -> ServiceRepository:
    return ServiceRepository(pool)
