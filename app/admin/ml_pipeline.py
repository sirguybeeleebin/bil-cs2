from django.contrib import admin

from app.models.ml_pipeline import MLPipeline


@admin.register(MLPipeline)
class MLPipelineAdmin(admin.ModelAdmin):
    list_display = ("pipeline_id", "path_to_pipeline_file", "created_at", "updated_at")
    search_fields = ("path_to_pipeline_file",)
    ordering = ("created_at",)
