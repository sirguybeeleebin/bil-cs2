import uuid

from django.db import models


class MLPipeline(models.Model):
    pipeline_id = models.UUIDField(default=uuid.uuid4, primary_key=True, editable=False)
    path_to_pipeline_file = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MLPipeline {self.pipeline_id}"

    class Meta:
        db_table = "ml_pipeline"
