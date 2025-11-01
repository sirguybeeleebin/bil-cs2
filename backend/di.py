from backend.repositories import (
    MapRepository,
    PlayerRepository,
    TeamRepository,
    TrainMetricRepository,
    TrainResultRepository,
    TrainTestSplitRepository,
)
from backend.services import DictionaryService, ForecasterService
from backend.tasks import inference_trained_model

map_repository = MapRepository()
team_repository = TeamRepository()
player_repository = PlayerRepository()
train_test_split_repository = TrainTestSplitRepository()
train_result_repository = TrainResultRepository()
train_metric_repository = TrainMetricRepository()

dictionary_service = DictionaryService(
    map_repository=map_repository,
    team_repository=team_repository,
    player_repository=player_repository,
)

forecaster_service = ForecasterService(inference_model_task=inference_trained_model)
