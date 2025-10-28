# backend/init_handlers.py
from backend.handlers import (
    make_forecast_handler,
    make_map_search_handler,
    make_player_search_handler,
    make_team_search_handler,
)
from backend.repositories import (
    MapRepository,
    MLPipelineMetricRepository,
    MLPipelineRepository,
    PlayerRepository,
    TeamRepository,
)

map_repo = MapRepository()
team_repo = TeamRepository()
player_repo = PlayerRepository()
ml_result_repo = MLPipelineRepository()
ml_result_metrics_repo = MLPipelineMetricRepository()


map_search_handler = make_map_search_handler(map_repo)
team_search_handler = make_team_search_handler(team_repo)
player_search_handler = make_player_search_handler(player_repo)
forecast_handler = make_forecast_handler()
