from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from dictionaries.repositories.map import MapRepository
from dictionaries.routers.map import make_map_router


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture
def mock_map_repo():
    repo = AsyncMock(spec=MapRepository)
    # Return dicts including created_at and updated_at
    repo.get_by_name.return_value = {
        "map_id": 1,
        "name": "Test Map",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    repo.search_by_name.return_value = [
        {
            "map_id": 1,
            "name": "Alpha Map",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        {
            "map_id": 2,
            "name": "Beta Map",
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
        },
    ]
    repo.upsert.return_value = True
    return repo


@pytest.fixture
def app(mock_map_repo):
    fastapi_app = FastAPI()
    router = make_map_router(mock_map_repo)
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


# ------------------------
# Tests
# ------------------------
def test_get_map_by_name_found(client, mock_map_repo):
    response = client.get("/maps/name/Test Map")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "map_id": 1,
        "name": "Test Map",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    mock_map_repo.get_by_name.assert_awaited_once_with("Test Map")


def test_get_map_by_name_not_found(client, mock_map_repo):
    mock_map_repo.get_by_name.return_value = None
    response = client.get("/maps/name/Unknown Map")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "Карта не найдена"}
    mock_map_repo.get_by_name.assert_awaited_once_with("Unknown Map")


def test_search_maps(client, mock_map_repo):
    response = client.get("/maps/search?q=Map&limit=10")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == [
        {
            "map_id": 1,
            "name": "Alpha Map",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        },
        {
            "map_id": 2,
            "name": "Beta Map",
            "created_at": "2025-01-02T00:00:00Z",
            "updated_at": "2025-01-02T00:00:00Z",
        },
    ]
    mock_map_repo.search_by_name.assert_awaited_once_with("Map", 10)


def test_save_maps(client, mock_map_repo):
    payload = [
        {"map_id": 1, "name": "Alpha Map", "created_at": "2025-01-01T00:00:00Z"},
        {"map_id": 2, "name": "Beta Map", "created_at": "2025-01-02T00:00:00Z"},
    ]
    response = client.post("/maps/save", json=payload)
    assert response.status_code == status.HTTP_200_OK

    # Check what was actually passed to upsert
    args, _ = mock_map_repo.upsert.await_args
    # Remove updated_at added by FastAPI default None if needed
    for d in args[0]:
        d.pop("updated_at", None)
    assert args[0] == payload
