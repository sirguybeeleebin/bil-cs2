from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from backend.models import TrainResult, TrainTestSplit
from backend.repositories import (
    MapRepository,
    PlayerRepository,
    TeamRepository,
    TrainMetricRepository,
    TrainResultRepository,
    TrainTestSplitRepository,
)


@pytest.mark.django_db
def test_map_repository_save_and_search():
    repo = MapRepository()
    saved = repo.save(map_id=1, name="Test Map")
    assert saved is not None
    assert saved["map_id"] == 1
    assert saved["name"] == "Test Map"

    results = repo.search_by_name(name="Test", limit=10, offset=0)
    assert len(results) == 1
    assert results[0]["name"] == "Test Map"


@pytest.mark.django_db
def test_team_repository_save_and_search():
    repo = TeamRepository()
    repo.save(team_id=1, name="Team A")
    results = repo.search_by_name(name="Team", limit=10, offset=0)
    assert len(results) == 1
    assert results[0]["name"] == "Team A"


@pytest.mark.django_db
def test_player_repository_save_and_search():
    repo = PlayerRepository()
    repo.save(player_id=1, name="Player One")
    results = repo.search_by_name(name="Player", limit=10, offset=0)
    assert len(results) == 1
    assert results[0]["name"] == "Player One"


@pytest.mark.django_db
def test_train_test_split_repository_save():
    repo = TrainTestSplitRepository()
    split_hash = "hash123"
    begin_min = datetime.now() - timedelta(days=1)
    begin_max = datetime.now()
    saved = repo.save(
        train_test_split_hash=split_hash,
        game_ids_train=[1, 2],
        game_ids_test=[3, 4],
        begin_at_min=begin_min,
        begin_at_max=begin_max,
    )
    assert saved is not None
    assert saved["train_test_split_hash"] == split_hash
    assert saved["game_ids_train"] == [1, 2]
    assert saved["game_ids_test"] == [3, 4]
    assert saved["begin_at_min"] == begin_min
    assert saved["begin_at_max"] == begin_max


@pytest.mark.django_db
def test_train_result_repository_save_and_get_last():
    _ = TrainTestSplit.objects.create(
        train_test_split_hash="split1", game_ids_train=[1], game_ids_test=[2]
    )
    repo = TrainResultRepository()
    train_result_id = uuid4()
    saved = repo.save(
        train_result_id=train_result_id,
        train_test_split_hash="split1",
        path_to_model="/path/to/model",
    )
    assert saved is not None
    assert saved["train_result_id"] == train_result_id

    last = repo.get_last()
    assert last is not None
    assert last["train_result_id"] == train_result_id


@pytest.mark.django_db
def test_train_metric_repository_save():
    split = TrainTestSplit.objects.create(
        train_test_split_hash="split2", game_ids_train=[1], game_ids_test=[2]
    )
    train_result = TrainResult.objects.create(
        train_result_id=uuid4(), train_test_split=split, path_to_model="/model"
    )
    repo = TrainMetricRepository()
    metric_id = uuid4()
    saved = repo.save(
        train_metric_id=metric_id,
        train_result_id=train_result.train_result_id,
        auc=0.9,
        f1=0.8,
        precision=0.85,
        recall=0.75,
        accuracy=0.88,
        tp=10,
        tn=20,
        fp=5,
        fn=2,
    )
    assert saved is not None
    assert saved["train_metric_id"] == metric_id
    assert saved["auc"] == 0.9
    assert saved["tp"] == 10
    assert saved["fn"] == 2
