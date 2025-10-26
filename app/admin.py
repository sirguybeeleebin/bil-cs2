from django.contrib import admin
from .models import Map, Team, Player, MLPipeline, MLPipelineMetrics, Prediction


@admin.register(Map)
class MapAdmin(admin.ModelAdmin):
    list_display = ("map_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("team_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("player_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(MLPipeline)
class MLPipelineAdmin(admin.ModelAdmin):
    list_display = ("pipeline_id", "path_to_pipeline_file", "created_at", "updated_at")
    search_fields = ("path_to_pipeline_file",)
    ordering = ("created_at",)


@admin.register(MLPipelineMetrics)
class MLPipelineMetricsAdmin(admin.ModelAdmin):
    list_display = ("metrics_id", "pipeline", "path_to_metrics_file", "created_at", "updated_at")
    search_fields = ("path_to_metrics_file", "pipeline__pipeline_id")
    ordering = ("created_at",)


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "prediction_id",
        "map",
        "team1",
        "team2",
        "team1_win_probability",
        "status",
        "ml_pipeline",
        "created_at",
        "updated_at",
    )
    search_fields = (
        "map__name",
        "team1__name",
        "team2__name",
        "ml_pipeline__pipeline_id",
    )
    list_filter = ("status", "map", "team1", "team2")
    ordering = ("created_at",)
    filter_horizontal = ("team1_players", "team2_players")
