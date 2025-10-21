from django.db import models

class MLPipeline(models.Model):
    class Status(models.TextChoices):
        TRAINING = "training", "Training"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    ml_pipeline_id = models.AutoField(primary_key=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.TRAINING,
    )
    pipeline_path = models.FilePathField(
        path="data/ml/",
        match=r".*\.pkl$",
        recursive=True,
        allow_files=True,
        allow_folders=False,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.ml_pipeline_id} ({self.status})"

    class Meta:
        verbose_name = "ML Pipeline"
        verbose_name_plural = "ML Pipelines"



