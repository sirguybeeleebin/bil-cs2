import uuid

from django.db import models


class Map(models.Model):
    map_id = models.BigIntegerField(unique=True)
    name = models.CharField(max_length=255)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (ID: {self.map_id})"


class Team(models.Model):
    team_id = models.BigIntegerField(unique=True)
    name = models.CharField(max_length=255)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (ID: {self.team_id})"


class Player(models.Model):
    player_id = models.BigIntegerField(unique=True)
    name = models.CharField(max_length=255)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} (ID: {self.player_id})"


class MLPipeline(models.Model):
    ml_pipeline_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    pipeline_file = models.FileField(upload_to="ml_results")
    metrics_file = models.FileField(upload_to="ml_results")

    def __str__(self):
        return f"MLPipeline {self.ml_pipeline_id}"


class MLPipelineMetric(models.Model):
    ml_pipeline = models.OneToOneField(
        MLPipeline, on_delete=models.CASCADE, related_name="metrics"
    )
    roc_auc = models.FloatField()
    f1 = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    accuracy = models.FloatField()
    tp = models.IntegerField()
    tn = models.IntegerField()
    fp = models.IntegerField()
    fn = models.IntegerField()

    def __str__(self):
        return f"Metrics for MLPipeline {self.ml_pipeline.ml_pipeline_id}"
