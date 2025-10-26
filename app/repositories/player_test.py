import uuid

import pytest
from django.core.management import call_command

from app.models.player import Player
from app.repositories.player import PlayerRepository


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(django_db_setup, django_db_blocker):
    """
    Автоматически применяет все миграции перед запуском тестов.
    """
    with django_db_blocker.unblock():
        call_command("migrate", run_syncdb=True)


@pytest.mark.django_db
def test_player_upsert_creates_new():
    repo = PlayerRepository()
    player_id = uuid.uuid4()
    data = {"player_id": player_id, "name": "Player 1"}

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["player_id"] == player_id
    obj = Player.objects.get(player_id=player_id)
    assert obj.name == "Player 1"


@pytest.mark.django_db
def test_player_upsert_updates_existing():
    player_id = uuid.uuid4()
    Player.objects.create(player_id=player_id, name="Old Player")
    repo = PlayerRepository()
    data = {"player_id": player_id, "name": "New Player"}

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["name"] == "New Player"
    obj = Player.objects.get(player_id=player_id)
    assert obj.name == "New Player"
