from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field
from starlette import status

from dictionaries.repositories.team import TeamRepository


class Team(BaseModel):
    team_id: int = Field(..., description="Уникальный идентификатор команды")
    name: str = Field(..., description="Название команды")
    created_at: str | None = Field(None, description="Дата создания")
    updated_at: str | None = Field(None, description="Дата обновления")


def make_team_router(team_repository: TeamRepository):
    router = APIRouter(prefix="/teams", tags=["teams"])

    @router.get(
        "/search", response_model=list[Team], summary="Поиск команд по части имени"
    )
    async def search_teams(
        q: str = Query(..., description="Часть имени для поиска"),
        limit: int = Query(10, description="Лимит результатов"),
    ):
        results = await team_repository.search_by_name(q, limit)
        return [Team(**r) for r in results]

    @router.get(
        "/name/{name}",
        response_model=Team,
        summary="Получить команду по точному имени",
    )
    async def get_team_by_name(name: str):
        result = await team_repository.get_by_name(name)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Команда не найдена"
            )
        return Team(**result)

    @router.post(
        "/save",
        status_code=status.HTTP_200_OK,
        summary="Сохранить или обновить список команд",
    )
    async def save_teams(
        data: list[Team] = Body(..., description="Список команд для сохранения"),
    ):
        await team_repository.upsert([d.dict() for d in data])
        return

    return router
