from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dictionaries.repositories.player import PlayerRepository
from dictionaries.routers.player import make_player_router


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture
def mock_player_repo():
    repo = AsyncMock(spec=PlayerRepository)
    # Return dicts including created_at and updated_at
    repo.get_by_name.return_value = {
        "player_id": 1,
        "name": "Test Player",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    repo.search_by_name.return_value = [
        {
            "player_id": 1,
            "name": "Alpha Player",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        {
            "player_id": 2,
            "name": "Beta Player",
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
        },
    ]
    repo.upsert.return_value = True
    return repo


@pytest.fixture
def app(mock_player_repo):
    fastapi_app = FastAPI()
    router = make_player_router(mock_player_repo)
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


# ------------------------
# Tests
# ------------------------
def test_get_player_by_name_found(client, mock_player_repo):
    response = client.get("/players/name/Test Player")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "player_id": 1,
        "name": "Test Player",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    mock_player_repo.get_by_name.assert_awaited_once_with("Test Player")


def test_get_player_by_name_not_found(client, mock_player_repo):
    mock_player_repo.get_by_name.return_value = None
    response = client.get("/players/name/Unknown Player")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "Игрок не найден"}
    mock_player_repo.get_by_name.assert_awaited_once_with("Unknown Player")


def test_search_players(client, mock_player_repo):
    response = client.get("/players/search?q=Player&limit=10")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [
        {
            "player_id": 1,
            "name": "Alpha Player",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        {
            "player_id": 2,
            "name": "Beta Player",
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
        },
    ]
    mock_player_repo.search_by_name.assert_awaited_once_with("Player", 10)


def test_save_players(client, mock_player_repo):
    payload = [
        {"player_id": 1, "name": "Alpha Player", "created_at": "2025-01-01T00:00:00Z"},
        {"player_id": 2, "name": "Beta Player", "created_at": "2025-01-02T00:00:00Z"},
    ]
    response = client.post("/players/save", json=payload)
    assert response.status_code == status.HTTP_200_OK

    # Check what was actually passed to upsert
    args, _ = mock_player_repo.upsert.await_args
    # Remove updated_at added by FastAPI default None if needed
    for d in args[0]:
        d.pop("updated_at", None)
    assert args[0] == payload
