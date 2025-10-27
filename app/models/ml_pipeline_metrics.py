import uuid

from django.db import models

from app.models.ml_pipeline import MLPipeline


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
