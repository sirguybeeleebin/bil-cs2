from django.contrib import admin

from .models import Map, MLPipeline, MLPipelineMetric, Player, Team


@admin.register(Map)
class MapAdmin(admin.ModelAdmin):
    list_display = ("map_id", "name", "updated_at")
    search_fields = ("map_id", "name")
    ordering = ("map_id",)


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("team_id", "name", "updated_at")
    search_fields = ("team_id", "name")
    ordering = ("team_id",)


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("player_id", "name", "updated_at")
    search_fields = ("player_id", "name")
    ordering = ("player_id",)


class MLPipelineMetricInline(admin.StackedInline):
    model = MLPipelineMetric
    readonly_fields = (
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
    )
    can_delete = False
    extra = 0


@admin.register(MLPipeline)
class MLPipelineAdmin(admin.ModelAdmin):
    list_display = ("ml_pipeline_id", "created_at", "pipeline_file", "metrics_file")
    readonly_fields = ("ml_pipeline_id", "created_at")
    ordering = ("-created_at",)
    inlines = [MLPipelineMetricInline]


@admin.register(MLPipelineMetric)
class MLPipelineMetricAdmin(admin.ModelAdmin):
    list_display = ("ml_pipeline", "roc_auc", "f1", "precision", "recall", "accuracy")
    readonly_fields = (
        "ml_pipeline",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
    )
    ordering = ("ml_pipeline",)
