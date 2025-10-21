# app/repositories/player_test.py

import os
import django
import pytest

# 1️⃣ Configure Django settings first
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# 2️⃣ Now import Django models and repositories
from app.models.player import Player
from app.repositories.player import PlayerRepository, make_player_repository

@pytest.mark.django_db
def test_upsert_creates_player():
    repo = make_player_repository()
    data = {"player_id": 1, "name": "Test Player"}
    
    result = repo.upsert(data)
    
    assert result is not None
    assert result["player_id"] == 1
    assert result["name"] == "Test Player"

@pytest.mark.django_db
def test_get_by_name_returns_player():
    Player.objects.create(player_id=2, name="Another Player")
    repo = make_player_repository()
    
    result = repo.get_by_name("Another Player")
    
    assert result is not None
    assert result["player_id"] == 2
    assert result["name"] == "Another Player"

@pytest.mark.django_db
def test_get_by_name_returns_none_for_missing():
    repo = make_player_repository()
    
    result = repo.get_by_name("Missing Player")
    
    assert result is None
