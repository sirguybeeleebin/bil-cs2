from pydantic import BaseModel, Field

class PlayerResponse(BaseModel):
    id: int = Field(..., title="ID игрока", description="Уникальный идентификатор игрока")
    name: str = Field(..., title="Имя игрока", description="Имя игрока")
    team_id: int = Field(..., title="ID команды", description="ID команды, к которой принадлежит игрок")