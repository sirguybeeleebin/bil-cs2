import uuid

import pytest
from django.core.management import call_command

from app.models.team import Team
from app.repositories.team import TeamRepository


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(django_db_setup, django_db_blocker):
    """
    Автоматически применяет все миграции перед запуском тестов.
    """
    with django_db_blocker.unblock():
        call_command("migrate", run_syncdb=True)


@pytest.mark.django_db
def test_team_upsert_creates_new():
    repo = TeamRepository()
    team_id = uuid.uuid4()
    data = {"team_id": team_id, "name": "Team A"}

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["team_id"] == team_id
    obj = Team.objects.get(team_id=team_id)
    assert obj.name == "Team A"


@pytest.mark.django_db
def test_team_upsert_updates_existing():
    team_id = uuid.uuid4()
    Team.objects.create(team_id=team_id, name="Old Team")
    repo = TeamRepository()
    data = {"team_id": team_id, "name": "New Team"}

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["name"] == "New Team"
    obj = Team.objects.get(team_id=team_id)
    assert obj.name == "New Team"
