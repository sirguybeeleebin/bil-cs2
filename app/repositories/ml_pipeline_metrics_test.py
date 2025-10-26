import uuid

import pytest
from django.core.management import call_command

from app.models.ml_pipeline import MLPipeline
from app.models.ml_pipeline_metrics import MLPipelineMetrics
from app.repositories.ml_pipeline_metrics import MLPipelineMetricsRepository


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(django_db_setup, django_db_blocker):
    """
    Автоматически применяет все миграции перед запуском тестов.
    """
    with django_db_blocker.unblock():
        call_command("migrate", run_syncdb=True)


@pytest.mark.django_db
def test_metrics_upsert_creates_new():
    repo = MLPipelineMetricsRepository()
    pipeline = MLPipeline.objects.create(path_to_pipeline_file="/tmp/pipeline.pkl")
    metrics_id = uuid.uuid4()
    data = {
        "metrics_id": metrics_id,
        "pipeline": pipeline,
        "path_to_metrics_file": "/tmp/metrics.pkl",
    }

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["metrics_id"] == metrics_id
    obj = MLPipelineMetrics.objects.get(metrics_id=metrics_id)
    assert obj.path_to_metrics_file == "/tmp/metrics.pkl"


@pytest.mark.django_db
def test_metrics_upsert_updates_existing():
    repo = MLPipelineMetricsRepository()
    pipeline = MLPipeline.objects.create(path_to_pipeline_file="/tmp/pipeline.pkl")
    metrics_id = uuid.uuid4()

    MLPipelineMetrics.objects.create(
        metrics_id=metrics_id,
        pipeline=pipeline,
        path_to_metrics_file="/old/metrics.pkl",
    )

    data = {
        "metrics_id": metrics_id,
        "pipeline": pipeline,
        "path_to_metrics_file": "/new/metrics.pkl",
    }

    result = repo.upsert(data.copy())

    assert result is not None
    obj = MLPipelineMetrics.objects.get(metrics_id=metrics_id)
    assert obj.path_to_metrics_file == "/new/metrics.pkl"
