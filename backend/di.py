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
