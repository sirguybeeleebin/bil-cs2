import uuid
from unittest.mock import MagicMock

import pytest
from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from app.handlers.forecast import make_forecast_handler

User = get_user_model()  # Correct way to get Django user model


@pytest.mark.django_db
def test_forecast_handler_creates_forecast_and_calls_task():
    # Mocks
    mock_repo = MagicMock()
    mock_task = MagicMock()

    forecast_id = uuid.uuid4()
    mock_repo.upsert.return_value = {
        "prediction_id": forecast_id,
        "ml_forecast_id": forecast_id,
        "team1_win_probability": 0.6,
        "team2_win_probability": 0.4,
        "created_at": "2025-10-27T12:00:00Z",
    }

    # Create API view with DI
    ForecastHandler = make_forecast_handler(mock_repo, mock_task)
    view = ForecastHandler.as_view()

    # Create test request
    factory = APIRequestFactory()
    user = User.objects.create_user(username="testuser", password="password")
    payload = {
        "map_id": str(uuid.uuid4()),
        "team1_id": str(uuid.uuid4()),
        "team2_id": str(uuid.uuid4()),
        "start_ct_team_id": str(uuid.uuid4()),
        "team1_player1_id": str(uuid.uuid4()),
        "team1_player2_id": str(uuid.uuid4()),
        "team1_player3_id": str(uuid.uuid4()),
        "team1_player4_id": str(uuid.uuid4()),
        "team1_player5_id": str(uuid.uuid4()),
        "team2_player1_id": str(uuid.uuid4()),
        "team2_player2_id": str(uuid.uuid4()),
        "team2_player3_id": str(uuid.uuid4()),
        "team2_player4_id": str(uuid.uuid4()),
        "team2_player5_id": str(uuid.uuid4()),
    }
    request = factory.post("/forecast/", payload, format="json")
    force_authenticate(request, user=user)

    # Call view
    response = view(request)
    data = response.data

    # Assertions
    assert response.status_code == status.HTTP_200_OK
    assert data["ml_forecast_id"] == str(forecast_id)
    assert data["status"] == "inference_started"
    assert data["team1_win_probability"] == 0.6
    assert data["team2_win_probability"] == 0.4

    mock_repo.upsert.assert_called_once()
    mock_task.delay.assert_called_once_with(str(forecast_id))


@pytest.mark.django_db
def test_forecast_handler_validation_error_same_teams():
    mock_repo = MagicMock()
    mock_task = MagicMock()
    ForecastHandler = make_forecast_handler(mock_repo, mock_task)
    view = ForecastHandler.as_view()

    factory = APIRequestFactory()
    user = User.objects.create_user(username="testuser", password="password")

    team_id = str(uuid.uuid4())
    payload = {
        "map_id": str(uuid.uuid4()),
        "team1_id": team_id,
        "team2_id": team_id,  # same ID triggers validation
        "start_ct_team_id": str(uuid.uuid4()),
        "team1_player1_id": str(uuid.uuid4()),
        "team1_player2_id": str(uuid.uuid4()),
        "team1_player3_id": str(uuid.uuid4()),
        "team1_player4_id": str(uuid.uuid4()),
        "team1_player5_id": str(uuid.uuid4()),
        "team2_player1_id": str(uuid.uuid4()),
        "team2_player2_id": str(uuid.uuid4()),
        "team2_player3_id": str(uuid.uuid4()),
        "team2_player4_id": str(uuid.uuid4()),
        "team2_player5_id": str(uuid.uuid4()),
    }
    request = factory.post("/forecast/", payload, format="json")
    force_authenticate(request, user=user)

    response = view(request)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Команды должны быть разными" in str(response.data)
    mock_repo.upsert.assert_not_called()
    mock_task.delay.assert_not_called()
