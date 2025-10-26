import uuid
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.tasks.ml_pipeline import make_ml_pipeline_task


@pytest.fixture
def mock_repositories():
    ml_pipeline_repo = MagicMock()
    ml_pipeline_metrics_repo = MagicMock()
    ml_pipeline_repo.upsert.return_value = True
    ml_pipeline_metrics_repo.upsert.return_value = True
    return ml_pipeline_repo, ml_pipeline_metrics_repo


@pytest.fixture
def mock_run_ml_pipeline_fn():
    def _fn(**kwargs):
        return "mock_pipeline", {"accuracy": 0.95}

    return _fn


@pytest.mark.django_db
def test_ml_pipeline_task_repo_upsert_failure(
    mock_repositories, mock_run_ml_pipeline_fn
):
    ml_pipeline_repo, ml_pipeline_metrics_repo = mock_repositories
    ml_pipeline_repo.upsert.return_value = False  # Simulate failure

    task = make_ml_pipeline_task(
        ml_pipeline_repo, ml_pipeline_metrics_repo, mock_run_ml_pipeline_fn
    )

    with (
        patch("builtins.open", mock_open()),
        patch("joblib.dump"),
        patch(
            "uuid.uuid4",
            side_effect=[
                uuid.UUID("12345678123456781234567812345678"),  # pipeline_id
                uuid.UUID("87654321876543218765432187654321"),  # metrics_id
            ],
        ),
        patch("config.settings.PATH_TO_ML_RESULTS_DIR", new="test_results"),
    ):
        result = task("fake_dir")

    assert result["status"] == "ошибка"
    assert "не удалось" in result["message"].lower()


@pytest.mark.django_db
def test_ml_pipeline_task_metrics_upsert_failure(
    mock_repositories, mock_run_ml_pipeline_fn
):
    ml_pipeline_repo, ml_pipeline_metrics_repo = mock_repositories
    ml_pipeline_metrics_repo.upsert.return_value = False  # Simulate metrics failure

    task = make_ml_pipeline_task(
        ml_pipeline_repo, ml_pipeline_metrics_repo, mock_run_ml_pipeline_fn
    )

    with (
        patch("builtins.open", mock_open()),
        patch("joblib.dump"),
        patch(
            "uuid.uuid4",
            side_effect=[
                uuid.UUID("12345678123456781234567812345678"),  # pipeline_id
                uuid.UUID("87654321876543218765432187654321"),  # metrics_id
                uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),  # extra if needed
            ],
        ),
        patch("config.settings.PATH_TO_ML_RESULTS_DIR", new="test_results"),
    ):
        result = task("fake_dir")

    assert result["status"] == "ошибка"
    assert "не удалось" in result["message"].lower()
