import json
from typing import List

import numpy as np
import redis.asyncio as redis
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field


class WinProbabilityRequest(BaseModel):
    map_id: int
    team1_id: int
    team2_id: int
    team1_player_ids: List[int] = Field(..., min_items=5, max_items=5)
    team2_player_ids: List[int] = Field(..., min_items=5, max_items=5)


class WinProbabilityResponse(BaseModel):
    team1_id: int
    team2_id: int
    team1_win_probability: float
    team2_win_probability: float


def make_forecast_router(redis_client: redis.Redis):
    router = APIRouter(prefix="/forecast", tags=["forecast"])

    @router.post("/get_team1_win_probability", response_model=WinProbabilityResponse)
    async def get_team1_win_probability(
        request: Request,
        win_probability_request: WinProbabilityRequest,
    ):
        predictor = request.app.state.predictor

        if predictor is None:
            team1_prob = 0.5
        else:
            t1_id, t2_id = sorted(
                [win_probability_request.team1_id, win_probability_request.team2_id]
            )
            X = np.array(
                [win_probability_request.map_id]
                + [t1_id, t2_id]
                + sorted(win_probability_request.team1_player_ids)
                + sorted(win_probability_request.team2_player_ids)
            ).reshape(1, -1)
            try:
                team1_prob = predictor.predict_proba(X)[0][1]
            except Exception:
                team1_prob = 0.5

        response = WinProbabilityResponse(
            team1_id=win_probability_request.team1_id,
            team2_id=win_probability_request.team2_id,
            team1_win_probability=team1_prob,
            team2_win_probability=1 - team1_prob,
        )

        event_data = {
            "request": win_probability_request.dict(),
            "response": response.dict(),
        }
        try:
            await redis_client.publish("forecast_events", json.dumps(event_data))
        except Exception as e:
            print(f"❌ Failed to publish forecast event to Redis: {e}")

        return response

    return router
