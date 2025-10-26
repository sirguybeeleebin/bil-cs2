import uuid

import pytest
from django.core.management import call_command

from app.models import Map, MLForecast, MLPipeline, Player, Team
from app.repositories.ml_forecast import MLForecastRepository


@pytest.fixture(scope="session", autouse=True)
def apply_migrations(django_db_setup, django_db_blocker):
    """
    Автоматически применяет все миграции перед запуском тестов.
    """
    with django_db_blocker.unblock():
        call_command("migrate", run_syncdb=True)


@pytest.mark.django_db
def test_forecast_upsert_creates_new():
    repo = MLForecastRepository()
    map_obj = Map.objects.create(name="Map 1")
    team1 = Team.objects.create(name="Team 1")
    team2 = Team.objects.create(name="Team 2")
    team_start = Team.objects.create(name="Team Start")
    player1 = Player.objects.create(name="Player 1")
    player2 = Player.objects.create(name="Player 2")
    pipeline = MLPipeline.objects.create(path_to_pipeline_file="/tmp/pipeline.pkl")

    ml_forecast_id = uuid.uuid4()
    data = {
        "ml_forecast_id": ml_forecast_id,
        "map": map_obj,
        "team1": team1,
        "team2": team2,
        "team1_players": [player1],
        "team2_players": [player2],
        "team_id_start_ct": team_start,
        "team1_win_probability": 0.6,
        "ml_pipeline": pipeline,
        "status": "PENDING",
    }

    result = repo.upsert(data.copy())
    assert result is not None
    assert result["ml_forecast_id"] == ml_forecast_id

    obj = MLForecast.objects.get(ml_forecast_id=ml_forecast_id)
    assert obj.team1_win_probability == 0.6
    assert list(obj.team1_players.all()) == [player1]
    assert list(obj.team2_players.all()) == [player2]


@pytest.mark.django_db
def test_forecast_upsert_updates_existing():
    map_obj = Map.objects.create(name="Map 1")
    team1 = Team.objects.create(name="Team 1")
    team2 = Team.objects.create(name="Team 2")
    team_start = Team.objects.create(name="Team Start")
    player1 = Player.objects.create(name="Player 1")
    player2 = Player.objects.create(name="Player 2")
    pipeline = MLPipeline.objects.create(path_to_pipeline_file="/tmp/pipeline.pkl")

    ml_forecast_id = uuid.uuid4()
    obj = MLForecast.objects.create(
        ml_forecast_id=ml_forecast_id,
        map=map_obj,
        team1=team1,
        team2=team2,
        team_id_start_ct=team_start,
        team1_win_probability=0.3,
    )
    obj.team1_players.add(player1)
    obj.team2_players.add(player2)

    repo = MLForecastRepository()
    data = {
        "ml_forecast_id": ml_forecast_id,
        "map": map_obj,
        "team1": team1,
        "team2": team2,
        "team1_players": [player1],
        "team2_players": [player2],
        "team_id_start_ct": team_start,
        "team1_win_probability": 0.9,
        "ml_pipeline": pipeline,
        "status": "COMPLETED",
    }

    result = repo.upsert(data.copy())
    assert result is not None

    obj.refresh_from_db()
    assert obj.team1_win_probability == 0.9
    assert obj.status == "COMPLETED"
    assert list(obj.team1_players.all()) == [player1]
    assert list(obj.team2_players.all()) == [player2]


@pytest.mark.django_db
def test_get_by_id_returns_forecast():
    map_obj = Map.objects.create(name="Map 1")
    team1 = Team.objects.create(name="Team 1")
    team2 = Team.objects.create(name="Team 2")
    team_start = Team.objects.create(name="Team Start")
    ml_forecast_id = uuid.uuid4()

    MLForecast.objects.create(
        ml_forecast_id=ml_forecast_id,
        map=map_obj,
        team1=team1,
        team2=team2,
        team_id_start_ct=team_start,
        team1_win_probability=0.5,
    )

    repo = MLForecastRepository()
    result = repo.get_by_id(ml_forecast_id)
    assert result is not None
    assert result["ml_forecast_id"] == ml_forecast_id


@pytest.mark.django_db
def test_get_by_id_returns_none_for_missing():
    repo = MLForecastRepository()
    result = repo.get_by_id(uuid.uuid4())
    assert result is None
