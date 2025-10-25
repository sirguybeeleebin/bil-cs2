from typing import Dict, List, Optional

import aiosqlite


class MLModelRepository:
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def init(self):
        """Создание таблицы, если не существует"""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                pipeline_path TEXT NOT NULL,
                metrics_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.commit()

    async def save_model(
        self, model_id: str, pipeline_path: str, metrics_path: str
    ) -> None:
        """Сохранить модель"""
        await self._db.execute(
            """
            INSERT INTO models (id, pipeline_path, metrics_path)
            VALUES (?, ?, ?)
            """,
            (model_id, pipeline_path, metrics_path),
        )
        await self._db.commit()

    async def list_models(self, limit: int = 100) -> List[Dict]:
        """Вернуть список моделей"""
        cursor = await self._db.execute(
            """
            SELECT id, pipeline_path, metrics_path, created_at
            FROM models
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "pipeline_path": r[1],
                "metrics_path": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

    async def get_latest_model(self) -> Optional[Dict]:
        """Вернуть последнюю модель"""
        models = await self.list_models(limit=1)
        return models[0] if models else None
