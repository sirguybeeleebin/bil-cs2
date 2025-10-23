from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport  # <-- use ASGI transport

from forecaster.routers.forecast import make_forecast_router


@pytest.fixture
def app():
    app = FastAPI()

    class DummyPredictor:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    app.state.predictor = DummyPredictor()
    mock_redis = AsyncMock()
    router = make_forecast_router(mock_redis)
    app.include_router(router)
    return app, mock_redis


@pytest.mark.asyncio
async def test_get_team1_win_probability_success(app):
    app_instance, mock_redis = app
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "map_id": 1,
            "team1_id": 100,
            "team2_id": 200,
            "team1_player_ids": [1, 2, 3, 4, 5],
            "team2_player_ids": [6, 7, 8, 9, 10],
        }
        response = await client.post(
            "/forecast/get_team1_win_probability", json=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert round(data["team1_win_probability"], 1) == 0.7
        assert round(data["team2_win_probability"], 1) == 0.3
        assert data["team1_id"] == 100
        assert data["team2_id"] == 200
        mock_redis.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_team1_win_probability_no_predictor(app):
    app_instance, mock_redis = app
    app_instance.state.predictor = None
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "map_id": 1,
            "team1_id": 100,
            "team2_id": 200,
            "team1_player_ids": [1, 2, 3, 4, 5],
            "team2_player_ids": [6, 7, 8, 9, 10],
        }
        response = await client.post(
            "/forecast/get_team1_win_probability", json=payload
        )
        assert response.status_code == 200
        data = response.json()
        assert round(data["team1_win_probability"], 1) == 0.5
        assert round(data["team2_win_probability"], 1) == 0.5
        mock_redis.publish.assert_awaited_once()
