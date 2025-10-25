import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

import asyncpg
import joblib
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from pydantic import BaseModel

# ========================================
# Load environment variables
# ========================================
load_dotenv()

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN", "postgresql://user:password@localhost:5432/ml_db"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "ml_pipeline_updates")


# ========================================
# Repository for storing models and metrics
# ========================================
class MLModelRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def save_model(self, model_id: str, pipeline_path: str, metrics_path: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO models(model_id, pipeline_path, metrics_path, created_at)
                VALUES($1, $2, $3, NOW())
                ON CONFLICT (model_id) DO UPDATE
                SET pipeline_path = EXCLUDED.pipeline_path,
                    metrics_path = EXCLUDED.metrics_path,
                    created_at = NOW()
                """,
                model_id,
                pipeline_path,
                metrics_path,
            )

    async def get_latest_model(self):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT model_id, pipeline_path, metrics_path FROM models ORDER BY created_at DESC LIMIT 1"
            )
            if row:
                return {
                    "model_id": row["model_id"],
                    "pipeline_path": row["pipeline_path"],
                    "metrics_path": row["metrics_path"],
                }
            return None

    async def get_model_by_id(self, model_id: str):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT model_id, pipeline_path, metrics_path FROM models WHERE model_id = $1",
                model_id,
            )
            if row:
                return {
                    "model_id": row["model_id"],
                    "pipeline_path": row["pipeline_path"],
                    "metrics_path": row["metrics_path"],
                }
            return None


# ========================================
# Pydantic models
# ========================================
class PredictionRequest(BaseModel):
    features: list[list[float]]  # X format for model
    model_id: str | None = None  # optional, if not provided use latest


class RedisMessage(BaseModel):
    model_id: str
    pipeline_path: str
    metrics_path: str
    extra: Dict = {}


# ========================================
# Redis consumer
# ========================================
async def redis_consumer(repo: MLModelRepository, redis_conn: aioredis.Redis):
    pubsub = redis_conn.pubsub()
    await pubsub.subscribe(REDIS_CHANNEL)

    async for message in pubsub.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            msg = RedisMessage(**data)
            await repo.save_model(msg.model_id, msg.pipeline_path, msg.metrics_path)
            print(f"Saved model {msg.model_id} from Redis")


# ========================================
# Lifespan context for FastAPI
# ========================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # PostgreSQL connection pool
    pool = await asyncpg.create_pool(dsn=POSTGRES_DSN)

    # Ensure table exists
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                pipeline_path TEXT NOT NULL,
                metrics_path TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)

    repo = MLModelRepository(pool)

    # Redis connection
    redis_conn = aioredis.from_url(REDIS_URL)

    # Start Redis consumer task
    redis_task = asyncio.create_task(redis_consumer(repo, redis_conn))

    # Inject into app.state
    app.state.pool = pool
    app.state.repo = repo
    app.state.redis_conn = redis_conn
    app.state.redis_task = redis_task

    try:
        yield
    finally:
        redis_task.cancel()
        await redis_conn.close()
        await pool.close()


# ========================================
# FastAPI app
# ========================================
app = FastAPI(title="CS:GO Winner Prediction API", lifespan=lifespan)


# Dependency to get repo
def get_repo(app: FastAPI = app) -> MLModelRepository:
    return app.state.repo


# ========================================
# Prediction endpoint
# ========================================
@app.post("/predict")
async def predict(
    request: PredictionRequest, repo: MLModelRepository = Depends(get_repo)
):
    if request.model_id:
        model_info = await repo.get_model_by_id(request.model_id)
        if not model_info:
            return {"error": f"Model {request.model_id} not found"}
    else:
        model_info = await repo.get_latest_model()
        if not model_info:
            return {"error": "No model available"}

    pipeline_path = model_info["pipeline_path"]
    pipeline = joblib.load(pipeline_path)

    predictions = pipeline.predict_proba(request.features)
    return {"model_id": model_info["model_id"], "predictions": predictions.tolist()}


# ========================================
# Health check endpoint
# ========================================
@app.get("/health")
async def health():
    return {"status": "ok"}
