# app/repositories/team_test.py

import os
import django
import pytest

# 1️⃣ Configure Django settings first
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# 2️⃣ Now import Django models and repositories
from app.models.team import Team
from app.repositories.team import TeamRepository, make_team_repository

@pytest.mark.django_db
def test_upsert_creates_team():
    repo = make_team_repository()
    data = {"team_id": 1, "name": "Test Team"}
    
    result = repo.upsert(data)
    
    assert result is not None
    assert result["team_id"] == 1
    assert result["name"] == "Test Team"

@pytest.mark.django_db
def test_get_by_name_returns_team():
    Team.objects.create(team_id=2, name="Another Team")
    repo = make_team_repository()
    
    result = repo.get_by_name("Another Team")
    
    assert result is not None
    assert result["team_id"] == 2
    assert result["name"] == "Another Team"

@pytest.mark.django_db
def test_get_by_name_returns_none_for_missing():
    repo = make_team_repository()
    
    result = repo.get_by_name("Missing Team")
    
    assert result is None
