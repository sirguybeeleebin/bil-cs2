import uuid

from django.db import models

from app.models.map import Map
from app.models.ml_pipeline import MLPipeline
from app.models.player import Player
from app.models.team import Team


class MLForecast(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "В ожидании"
        IN_PROGRESS = "IN_PROGRESS", "В процессе"
        COMPLETED = "COMPLETED", "Завершено"
        FAILED = "FAILED", "Ошибка"

    ml_forecast_id = models.UUIDField(
        default=uuid.uuid4, primary_key=True, editable=False
    )
    map = models.ForeignKey(Map, on_delete=models.PROTECT, related_name="forecasts")
    team1 = models.ForeignKey(
        Team, on_delete=models.PROTECT, related_name="team1_forecasts"
    )
    team2 = models.ForeignKey(
        Team, on_delete=models.PROTECT, related_name="team2_forecasts"
    )
    team1_players = models.ManyToManyField(Player, related_name="team1_forecasts")
    team2_players = models.ManyToManyField(Player, related_name="team2_forecasts")
    team_id_start_ct = models.ForeignKey(
        Team, on_delete=models.PROTECT, related_name="start_ct_forecasts"
    )
    team1_win_probability = models.FloatField(null=True, blank=True)
    ml_pipeline = models.ForeignKey(
        MLPipeline, on_delete=models.SET_NULL, null=True, blank=True
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Forecast {self.ml_prediction_id} ({self.status})"

    class Meta:
        db_table = "ml_forecast"
