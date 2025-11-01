from django.contrib import admin

from .models import Map, Player, Team, TrainMetric, TrainResult, TrainTestSplit


@admin.register(Map)
class MapAdmin(admin.ModelAdmin):
    list_display = ("map_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("map_id",)


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("team_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("team_id",)


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("player_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("player_id",)


@admin.register(TrainTestSplit)
class TrainTestSplitAdmin(admin.ModelAdmin):
    list_display = (
        "train_test_split_hash",
        "game_ids_train",
        "game_ids_test",
        "created_at",
        "updated_at",
    )
    search_fields = ("train_test_split_hash",)
    ordering = ("created_at",)


@admin.register(TrainResult)
class TrainResultAdmin(admin.ModelAdmin):
    list_display = (
        "train_result_id",
        "train_test_split",
        "path_to_model",
        "created_at",
        "updated_at",
    )
    search_fields = ("train_result_id", "path_to_model")
    list_filter = ("train_test_split",)
    ordering = ("created_at",)


@admin.register(TrainMetric)
class TrainMetricAdmin(admin.ModelAdmin):
    list_display = (
        "train_metric_id",
        "train_result",
        "auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
        "created_at",
        "updated_at",
    )
    search_fields = ("train_metric_id", "train_result__train_result_id")
    list_filter = ("train_result",)
    ordering = ("created_at",)
