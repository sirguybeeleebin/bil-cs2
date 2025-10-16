from pydantic import BaseModel, Field
from fastapi import APIRouter
from backend.services.player import PlayerService
from backend.services.forecast import ForecastService
from backend.models.player import PlayerResponse
from backend.models.forecast import ForecastRequest, ForecastResponse

class PlayerResponse(BaseModel):
    id: int = Field(..., title="ID игрока", description="Уникальный идентификатор игрока")
    name: str = Field(..., title="Имя игрока", description="Имя игрока")
    team_id: int = Field(..., title="ID команды", description="ID команды, к которой принадлежит игрок")
    



def create_player_router(player_service: PlayerService) -> APIRouter:
    router = APIRouter(prefix="/player")

    @router.get("/{player_name}", response_model=list[PlayerResponse])
    async def search_player_by_name(player_name: str):
        return await player_service.search_player_by_name(player_name)

    return router



