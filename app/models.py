import uuid

from django.db import models


class Map(models.Model):
    map_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "map"


class Team(models.Model):
    team_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "team"


class Player(models.Model):
    player_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = "player"


class MLPipeline(models.Model):
    pipeline_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    path_to_pipeline_file = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MLPipeline {self.pipeline_id}"

    class Meta:
        db_table = "ml_pipeline"


class MLPipelineMetrics(models.Model):
    metrics_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    pipeline = models.ForeignKey(
        MLPipeline, on_delete=models.CASCADE, related_name="metrics"
    )
    path_to_metrics_file = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MLPipelineMetrics {self.metrics_id}"

    class Meta:
        db_table = "ml_pipeline_metrics"


class Prediction(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "В ожидании"
        IN_PROGRESS = "IN_PROGRESS", "В процессе"
        COMPLETED = "COMPLETED", "Завершено"
        FAILED = "FAILED", "Ошибка"

    prediction_id = models.UUIDField(
        default=uuid.uuid4, primary_key=True, editable=False
    )
    map = models.ForeignKey(Map, on_delete=models.PROTECT, related_name="predictions")
    team1 = models.ForeignKey(
        Team, on_delete=models.PROTECT, related_name="team1_predictions"
    )
    team2 = models.ForeignKey(
        Team, on_delete=models.PROTECT, related_name="team2_predictions"
    )
    team1_players = models.ManyToManyField(Player, related_name="team1_predictions")
    team2_players = models.ManyToManyField(Player, related_name="team2_predictions")
    team_id_start_ct = models.ForeignKey(
        Team, on_delete=models.PROTECT, related_name="start_ct_predictions"
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
        return f"Prediction {self.prediction_id} ({self.status})"

    class Meta:
        db_table = "prediction"
