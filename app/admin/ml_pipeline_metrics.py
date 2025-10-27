from django.contrib import admin

from app.models.ml_pipeline_metrics import MLPipelineMetrics


@admin.register(MLPipelineMetrics)
class MLPipelineMetricsAdmin(admin.ModelAdmin):
    list_display = (
        "metrics_id",
        "pipeline",
        "path_to_metrics_file",
        "created_at",
        "updated_at",
    )
    search_fields = ("path_to_metrics_file",)
    ordering = ("created_at",)
    list_filter = ("pipeline",)
