from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    team1_id: int = Field(..., title="ID первой команды", description="Уникальный идентификатор первой команды")
    team2_id: int = Field(..., title="ID второй команды", description="Уникальный идентификатор второй команды")
    team1_player1_id: int = Field(..., title="Игрок 1 первой команды", description="ID первого игрока первой команды")
    team1_player2_id: int = Field(..., title="Игрок 2 первой команды", description="ID второго игрока первой команды")
    team1_player3_id: int = Field(..., title="Игрок 3 первой команды", description="ID третьего игрока первой команды")
    team1_player4_id: int = Field(..., title="Игрок 4 первой команды", description="ID четвертого игрока первой команды")
    team1_player5_id: int = Field(..., title="Игрок 5 первой команды", description="ID пятого игрока первой команды")
    team2_player1_id: int = Field(..., title="Игрок 1 второй команды", description="ID первого игрока второй команды")
    team2_player2_id: int = Field(..., title="Игрок 2 второй команды", description="ID второго игрока второй команды")
    team2_player3_id: int = Field(..., title="Игрок 3 второй команды", description="ID третьего игрока второй команды")
    team2_player4_id: int = Field(..., title="Игрок 4 второй команды", description="ID четвертого игрока второй команды")
    team2_player5_id: int = Field(..., title="Игрок 5 второй команды", description="ID пятого игрока второй команды")

class ForecastResponse(BaseModel):
    team1_id: int = Field(..., title="ID первой команды", description="Уникальный идентификатор первой команды")
    team2_id: int = Field(..., title="ID второй команды", description="Уникальный идентификатор второй команды")
    team1_win_prob: float = Field(..., title="Вероятность победы первой команды", description="Вероятность выигрыша первой команды (от 0 до 1)")
    team2_win_prob: float = Field(..., title="Вероятность победы второй команды", description="Вероятность выигрыша второй команды (от 0 до 1)")
    
    
def create_forecast_router(forecast_service: ForecastService) -> APIRouter:
    router = APIRouter(prefix="/forecast")

    @router.post("/", response_model=ForecastResponse)
    async def get_forecast(request: ForecastRequest):
        return await forecast_service.get_forecast(request)

    return router