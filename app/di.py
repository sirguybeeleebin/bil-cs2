from app.repositories import (
    MapRepository,
    MLPipelineMetricsRepository,
    MLPipelineRepository,
    PlayerRepository,
    PredictionRepository,
    TeamRepository,
)

map_repository = MapRepository()
team_repository = TeamRepository()
player_repository = PlayerRepository()
ml_pipeline_repository = MLPipelineRepository()
ml_pipeline_metrics_repository = MLPipelineMetricsRepository()
prediction_repository = PredictionRepository()
