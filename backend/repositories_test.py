from uuid import uuid4

import pytest

from backend.models import MLPipeline
from backend.repositories import (
    MapRepository,
    MLPipelineMetricRepository,
    MLPipelineRepository,
    PlayerRepository,
    TeamRepository,
)

pytestmark = pytest.mark.django_db


def test_map_upsert_and_search():
    repo = MapRepository()

    result = repo.upsert(map_id=1, name="Dust II")
    assert result["map_id"] == 1
    assert result["name"] == "Dust II"

    # обновление
    result2 = repo.upsert(map_id=1, name="Dust II Updated")
    assert result2["name"] == "Dust II Updated"

    results = repo.search_by_name("Dust")
    assert len(results) == 1
    assert results[0]["name"] == "Dust II Updated"


def test_team_upsert_and_search():
    repo = TeamRepository()

    result = repo.upsert(team_id=1, name="Astralis")
    assert result["team_id"] == 1
    assert result["name"] == "Astralis"

    results = repo.search_by_name("stra")
    assert len(results) == 1
    assert results[0]["name"] == "Astralis"


def test_player_upsert_and_search():
    repo = PlayerRepository()

    result = repo.upsert(player_id=1, name="s1mple")
    assert result["player_id"] == 1
    assert result["name"] == "s1mple"

    results = repo.search_by_name("s1m")
    assert len(results) == 1
    assert results[0]["name"] == "s1mple"


def test_ml_pipeline_upsert(tmp_path):
    repo = MLPipelineRepository()

    pipeline_file = tmp_path / "pipeline.pkl"
    pipeline_file.write_bytes(b"dummy pipeline data")

    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_bytes(b'{"roc_auc": 0.9}')

    pipeline_id = uuid4()
    result = repo.upsert(
        ml_pipeline_id=pipeline_id,
        pipeline_file_path=str(pipeline_file),
        metrics_file_path=str(metrics_file),
    )
    assert result["ml_pipeline_id"] == pipeline_id
    assert result["pipeline_file_path"] == str(pipeline_file)
    assert result["metrics_file_path"] == str(metrics_file)


def test_ml_pipeline_metric_upsert():
    pipeline = MLPipeline.objects.create(
        pipeline_file_path="pipeline.pkl", metrics_file_path="metrics.json"
    )
    repo = MLPipelineMetricRepository()
    metric_id = uuid4()

    result = repo.upsert(
        ml_pipeline_metric_id=metric_id,
        ml_pipeline_id=pipeline.ml_pipeline_id,
        roc_auc=0.9,
        f1=0.8,
        precision=0.7,
        recall=0.85,
        accuracy=0.95,
        tp=10,
        tn=20,
        fp=1,
        fn=2,
    )

    assert result["ml_pipeline_metric_id"] == metric_id
    assert str(result["ml_pipeline_id"]) == str(pipeline.ml_pipeline_id)
    assert result["roc_auc"] == 0.9
    assert result["f1"] == 0.8
    assert result["precision"] == 0.7
    assert result["recall"] == 0.85
    assert result["accuracy"] == 0.95
    assert result["tp"] == 10
    assert result["tn"] == 20
    assert result["fp"] == 1
    assert result["fn"] == 2
