# app/repositories/map_test.py

import os
import django
import pytest

# 1️⃣ Configure Django settings first
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# 2️⃣ Now import Django models and repositories
from app.models.map import Map
from app.repositories.map import MapRepository, make_map_repository

@pytest.mark.django_db
def test_upsert_creates_map():
    repo = make_map_repository()
    data = {"map_id": 1, "name": "Test Map"}
    
    result = repo.upsert(data)
    
    assert result is not None
    assert result["map_id"] == 1
    assert result["name"] == "Test Map"

@pytest.mark.django_db
def test_get_by_name_returns_map():
    Map.objects.create(map_id=2, name="Another Map")
    repo = make_map_repository()
    
    result = repo.get_by_name("Another Map")
    
    assert result is not None
    assert result["map_id"] == 2
    assert result["name"] == "Another Map"

@pytest.mark.django_db
def test_get_by_name_returns_none_for_missing():
    repo = make_map_repository()
    
    result = repo.get_by_name("Missing Map")
    
    assert result is None
