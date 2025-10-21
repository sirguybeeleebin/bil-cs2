from django.db import models


class MLPipeline(models.Model):
    class Status(models.TextChoices):
        TRAINING = "training", "Training"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    ml_pipeline_id = models.AutoField(primary_key=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.TRAINING,
    )
    pipeline_path = models.FilePathField(
        path="data/ml/",
        match=r".*\.pkl$",
        recursive=True,
        allow_files=True,
        allow_folders=False,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.ml_pipeline_id} ({self.status})"

    class Meta:
        verbose_name = "ML Pipeline"
        verbose_name_plural = "ML Pipelines"


class MLMetrics(models.Model):
    ml_metrics_id = models.AutoField(primary_key=True)
    pipeline = models.ForeignKey(
        MLPipeline, on_delete=models.CASCADE, related_name="metrics"
    )
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return (
            f"Metrics {self.ml_metrics_id} for pipeline {self.pipeline.ml_pipeline_id}"
        )

    class Meta:
        verbose_name = "ML Metric"
        verbose_name_plural = "ML Metrics"


class Map(models.Model):
    map_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.map_id})"

    class Meta:
        verbose_name = "Map"
        verbose_name_plural = "Maps"


class Team(models.Model):
    team_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.team_id})"

    class Meta:
        verbose_name = "Team"
        verbose_name_plural = "Teams"


class Player(models.Model):
    player_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.player_id})"

    class Meta:
        verbose_name = "Player"
        verbose_name_plural = "Players"
