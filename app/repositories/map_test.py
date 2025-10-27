import uuid

import pytest
from django.core.management import call_command

from app.models.map import Map
from app.repositories.map import MapRepository


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(django_db_setup, django_db_blocker):
    with django_db_blocker.unblock():
        call_command("migrate", run_syncdb=True)


@pytest.mark.django_db
def test_map_upsert_creates_new():
    repo = MapRepository()
    map_id = uuid.uuid4()
    data = {"map_id": map_id, "name": "Test Map"}

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["map_id"] == map_id
    obj = Map.objects.get(map_id=map_id)
    assert obj.name == "Test Map"


@pytest.mark.django_db
def test_map_upsert_updates_existing():
    map_id = uuid.uuid4()
    Map.objects.create(map_id=map_id, name="Old Name")
    repo = MapRepository()
    data = {"map_id": map_id, "name": "Updated Name"}

    result = repo.upsert(data.copy())

    assert result is not None
    assert result["name"] == "Updated Name"
    obj = Map.objects.get(map_id=map_id)
    assert obj.name == "Updated Name"
