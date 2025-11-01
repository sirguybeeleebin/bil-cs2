import uuid

from django.contrib.postgres.fields import ArrayField
from django.db import models


class Map(models.Model):
    map_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Team(models.Model):
    team_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Player(models.Model):
    player_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class TrainTestSplit(models.Model):
    train_test_split_hash = models.CharField(
        max_length=64, primary_key=True, editable=False
    )
    game_ids_train = ArrayField(models.IntegerField(), blank=True, default=list)
    game_ids_test = ArrayField(models.IntegerField(), blank=True, default=list)
    begin_at_min = models.DateTimeField(null=True, blank=True)
    begin_at_max = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.train_test_split_hash)


class TrainResult(models.Model):
    train_result_id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    train_test_split = models.ForeignKey(
        TrainTestSplit, on_delete=models.CASCADE, related_name="train_results"
    )
    path_to_model = models.CharField(max_length=1024)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.train_result_id)


class TrainMetric(models.Model):
    train_metric_id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    train_result = models.OneToOneField(
        TrainResult, on_delete=models.CASCADE, related_name="train_metric"
    )

    auc = models.FloatField(null=True, blank=True)
    f1 = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    tp = models.IntegerField(null=True, blank=True)
    tn = models.IntegerField(null=True, blank=True)
    fp = models.IntegerField(null=True, blank=True)
    fn = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.train_metric_id)
