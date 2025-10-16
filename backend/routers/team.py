from pydantic import BaseModel, Field

class TeamResponse(BaseModel):
    team_id: int = Field(..., title="ID команды", description="Уникальный идентификатор команды")
    name: str = Field(..., title="Название команды", description="Название команды")


