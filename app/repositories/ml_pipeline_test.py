import uuid

import pytest
from django.core.management import call_command

from app.models.ml_pipeline import MLPipeline
from app.repositories.ml_pipeline import MLPipelineRepository


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(django_db_setup, django_db_blocker):
    """
    Автоматически применяет все миграции перед запуском тестов.
    """
    with django_db_blocker.unblock():
        call_command("migrate", run_syncdb=True)


@pytest.mark.django_db
def test_pipeline_upsert_creates_new():
    repo = MLPipelineRepository()
    pipeline_id = uuid.uuid4()
    data = {
        "pipeline_id": pipeline_id,
        "path_to_pipeline_file": "/tmp/pipeline.pkl",
    }

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["pipeline_id"] == pipeline_id
    obj = MLPipeline.objects.get(pipeline_id=pipeline_id)
    assert obj.path_to_pipeline_file == "/tmp/pipeline.pkl"


@pytest.mark.django_db
def test_pipeline_upsert_updates_existing():
    pipeline_id = uuid.uuid4()
    MLPipeline.objects.create(
        pipeline_id=pipeline_id, path_to_pipeline_file="/old/path.pkl"
    )
    repo = MLPipelineRepository()
    data = {
        "pipeline_id": pipeline_id,
        "path_to_pipeline_file": "/new/path.pkl",
    }

    result = repo.upsert(data.copy())

    assert result is not None
    obj = MLPipeline.objects.get(pipeline_id=pipeline_id)
    assert obj.path_to_pipeline_file == "/new/path.pkl"
