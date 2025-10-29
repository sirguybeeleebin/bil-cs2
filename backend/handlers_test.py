from unittest.mock import Mock, patch

import pytest
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from backend.handlers import (
    ForecastHandler,
    MapSearchHandler,
    PlayerSearchHandler,
    TeamSearchHandler,
)


@pytest.fixture
def api_factory():
    return APIRequestFactory()


class DummyUser:
    is_authenticated = True


# ------------------------
# Map Search Handler
# ------------------------
def test_map_search_handler(api_factory):
    mock_repo = Mock()
    mock_repo.search_by_name.return_value = [{"id": 1, "name": "Dust2"}]

    view = MapSearchHandler.as_view()
    view.cls.repo = mock_repo  # подменяем репозиторий

    request = api_factory.get("/maps?name=Dust2&limit=5&offset=0")
    force_authenticate(request, user=DummyUser())
    response = view(request)

    mock_repo.search_by_name.assert_called_once_with(name="Dust2", limit=5, offset=0)
    assert response.status_code == status.HTTP_200_OK
    assert response.data == [{"id": 1, "name": "Dust2"}]


# ------------------------
# Team Search Handler
# ------------------------
def test_team_search_handler(api_factory):
    mock_repo = Mock()
    mock_repo.search_by_name.return_value = [{"id": 1, "name": "TeamA"}]

    view = TeamSearchHandler.as_view()
    view.cls.repo = mock_repo

    request = api_factory.get("/teams?name=TeamA")
    force_authenticate(request, user=DummyUser())
    response = view(request)

    mock_repo.search_by_name.assert_called_once_with(name="TeamA", limit=10, offset=0)
    assert response.status_code == status.HTTP_200_OK
    assert response.data == [{"id": 1, "name": "TeamA"}]


# ------------------------
# Player Search Handler
# ------------------------
def test_player_search_handler(api_factory):
    mock_repo = Mock()
    mock_repo.search_by_name.return_value = [{"id": 1, "name": "Player1"}]

    view = PlayerSearchHandler.as_view()
    view.cls.repo = mock_repo

    request = api_factory.get("/players?name=Player1&limit=2")
    force_authenticate(request, user=DummyUser())
    response = view(request)

    mock_repo.search_by_name.assert_called_once_with(name="Player1", limit=2, offset=0)
    assert response.status_code == status.HTTP_200_OK
    assert response.data == [{"id": 1, "name": "Player1"}]


# ------------------------
# Forecast Handler
# ------------------------
@patch("backend.handlers.ml_forecast_inference_task.apply_async")
def test_forecast_handler_success(mock_task, api_factory):
    mock_result = Mock()
    mock_result.id = "test-task-id"
    mock_task.return_value = mock_result

    view = ForecastHandler.as_view()

    request_data = {
        "map_id": 1,
        "team1_id": 1,
        "team2_id": 2,
        "team1_player1_id": 1,
        "team1_player2_id": 2,
        "team1_player3_id": 3,
        "team1_player4_id": 4,
        "team1_player5_id": 5,
        "team2_player1_id": 6,
        "team2_player2_id": 7,
        "team2_player3_id": 8,
        "team2_player4_id": 9,
        "team2_player5_id": 10,
    }

    request = api_factory.post("/forecast", request_data, format="json")
    force_authenticate(request, user=DummyUser())
    response = view(request)

    mock_task.assert_called_once()
    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.data["task_id"] == "test-task-id"


# ------------------------
# Forecast Handler: invalid data
# ------------------------
def test_forecast_handler_invalid_data(api_factory):
    view = ForecastHandler.as_view()

    request_data = {"map_id": "invalid"}  # недопустимые поля
    request = api_factory.post("/forecast", request_data, format="json")
    force_authenticate(request, user=DummyUser())
    response = view(request)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "team1_id" in response.data
    assert "team2_id" in response.data
