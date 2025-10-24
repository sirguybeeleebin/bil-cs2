from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.routers.auth import make_auth_router
from auth.services.auth import AuthService

JWT_TOKEN = "mocked-jwt-token"


@pytest.fixture
def auth_service_mock():
    service = AsyncMock(spec=AuthService)
    service.register_user.return_value = {
        "user_id": uuid4(),
        "username": "alice",
        "created_at": "2025-10-23T00:00:00Z",
        "updated_at": "2025-10-23T00:00:00Z",
    }
    service.authenticate_user.return_value = JWT_TOKEN
    service.register_service.return_value = {
        "service_id": uuid4(),
        "client_id": "etl_001",
        "created_at": "2025-10-23T00:00:00Z",
        "updated_at": "2025-10-23T00:00:00Z",
    }
    service.authenticate_service.return_value = JWT_TOKEN
    service.get_me.return_value = {
        "user_id": uuid4(),
        "username": "alice",
        "created_at": "2025-10-23T00:00:00Z",
        "updated_at": "2025-10-23T00:00:00Z",
    }
    return service


@pytest.fixture
def client(auth_service_mock):
    app = FastAPI()
    router = make_auth_router(auth_service_mock)
    app.include_router(router)
    return TestClient(app)


def test_register_user(client):
    response = client.post(
        "/auth/register", json={"username": "alice", "password": "StrongPass1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "alice"


def test_login_user(client):
    response = client.post(
        "/auth/login", json={"username": "alice", "password": "StrongPass1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] == JWT_TOKEN


def test_register_service(client):
    response = client.post(
        "/auth/service/register",
        json={"client_id": "etl_001", "client_secret": "secret123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == "etl_001"


def test_service_token(client):
    response = client.post(
        "/auth/service/token",
        json={"client_id": "etl_001", "client_secret": "secret123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] == JWT_TOKEN


def test_get_me_success(client, auth_service_mock):
    user_id = uuid4()
    auth_service_mock.get_me.return_value = {
        "user_id": str(user_id),
        "username": "testuser",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
    }

    response = client.get("/auth/me", headers={"user-id": str(user_id)})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == str(user_id)
