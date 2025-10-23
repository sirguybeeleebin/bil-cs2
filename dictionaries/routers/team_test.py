from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dictionaries.repositories.team import TeamRepository
from dictionaries.routers.team import make_team_router


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture
def mock_team_repo():
    repo = AsyncMock(spec=TeamRepository)
    # Return dicts including created_at and updated_at
    repo.get_by_name.return_value = {
        "team_id": 1,
        "name": "Test Team",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    repo.search_by_name.return_value = [
        {
            "team_id": 1,
            "name": "Alpha Team",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        {
            "team_id": 2,
            "name": "Beta Team",
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
        },
    ]
    repo.upsert.return_value = True
    return repo


@pytest.fixture
def app(mock_team_repo):
    fastapi_app = FastAPI()
    router = make_team_router(mock_team_repo)
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


# ------------------------
# Tests
# ------------------------
def test_get_team_by_name_found(client, mock_team_repo):
    response = client.get("/teams/name/Test Team")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "team_id": 1,
        "name": "Test Team",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    mock_team_repo.get_by_name.assert_awaited_once_with("Test Team")


def test_get_team_by_name_not_found(client, mock_team_repo):
    mock_team_repo.get_by_name.return_value = None
    response = client.get("/teams/name/Unknown Team")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "Команда не найдена"}
    mock_team_repo.get_by_name.assert_awaited_once_with("Unknown Team")


def test_search_teams(client, mock_team_repo):
    response = client.get("/teams/search?q=Team&limit=10")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [
        {
            "team_id": 1,
            "name": "Alpha Team",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        {
            "team_id": 2,
            "name": "Beta Team",
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
        },
    ]
    mock_team_repo.search_by_name.assert_awaited_once_with("Team", 10)


def test_save_teams(client, mock_team_repo):
    payload = [
        {"team_id": 1, "name": "Alpha Team", "created_at": "2025-01-01T00:00:00Z"},
        {"team_id": 2, "name": "Beta Team", "created_at": "2025-01-02T00:00:00Z"},
    ]
    response = client.post("/teams/save", json=payload)
    assert response.status_code == status.HTTP_200_OK

    # Check what was actually passed to upsert
    args, _ = mock_team_repo.upsert.await_args
    # Remove updated_at added by FastAPI default None if needed
    for d in args[0]:
        d.pop("updated_at", None)
    assert args[0] == payload
