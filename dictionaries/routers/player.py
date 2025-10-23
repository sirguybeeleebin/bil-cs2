from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field
from starlette import status

from dictionaries.repositories.player import PlayerRepository


class Player(BaseModel):
    player_id: int = Field(..., description="Уникальный идентификатор игрока")
    name: str = Field(..., description="Имя игрока")
    created_at: str | None = Field(None, description="Дата создания")
    updated_at: str | None = Field(None, description="Дата обновления")


def make_player_router(player_repository: PlayerRepository):
    router = APIRouter(prefix="/players", tags=["players"])

    @router.get(
        "/search", response_model=list[Player], summary="Поиск игроков по части имени"
    )
    async def search_players(
        q: str = Query(..., description="Часть имени для поиска"),
        limit: int = Query(10, description="Лимит результатов"),
    ):
        results = await player_repository.search_by_name(q, limit)
        return [Player(**r) for r in results]

    @router.get(
        "/name/{name}",
        response_model=Player,
        summary="Получить игрока по точному имени",
    )
    async def get_player_by_name(name: str):
        result = await player_repository.get_by_name(name)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Игрок не найден"
            )
        return Player(**result)

    @router.post(
        "/save",
        status_code=status.HTTP_200_OK,
        summary="Сохранить или обновить список игроков",
    )
    async def save_players(
        data: list[Player] = Body(..., description="Список игроков для сохранения"),
    ):
        await player_repository.upsert([d.dict() for d in data])
        return

    return router
