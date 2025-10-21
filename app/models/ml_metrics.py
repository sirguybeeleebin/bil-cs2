from django.db import models
from app.models.ml_pipeline import MLPipeline  

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
        return f"Metrics {self.ml_metrics_id} for pipeline {self.pipeline.ml_pipeline_id}"

    class Meta:
        verbose_name = "ML Metric"
        verbose_name_plural = "ML Metrics"