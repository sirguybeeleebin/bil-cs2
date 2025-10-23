from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field
from starlette import status

from dictionaries.repositories.map import MapRepository


class Map(BaseModel):
    map_id: int = Field(..., description="Уникальный идентификатор карты")
    name: str = Field(..., description="Название карты")
    created_at: str | None = Field(None, description="Дата создания")
    updated_at: str | None = Field(None, description="Дата обновления")


def make_map_router(map_repository: MapRepository):
    router = APIRouter(prefix="/maps", tags=["maps"])

    @router.get(
        "/search", response_model=list[Map], summary="Поиск карт по части имени"
    )
    async def search_maps(
        q: str = Query(..., description="Часть имени для поиска"),
        limit: int = Query(10, description="Лимит результатов"),
    ):
        results = await map_repository.search_by_name(q, limit)
        return [Map(**r) for r in results]

    @router.get(
        "/name/{name}",
        response_model=Map,
        summary="Получить карту по точному имени",
    )
    async def get_map_by_name(name: str):
        result = await map_repository.get_by_name(name)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Карта не найдена"
            )
        return Map(**result)

    @router.post(
        "/save",
        status_code=status.HTTP_200_OK,
        summary="Сохранить или обновить список карт",
    )
    async def save_maps(
        data: list[Map] = Body(..., description="Список карт для сохранения"),
    ):
        await map_repository.upsert([d.dict() for d in data])
        return

    return router
