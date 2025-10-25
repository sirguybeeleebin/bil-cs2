from django.db import models
import uuid

class Prediction(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    payload = models.JSONField()
    status = models.CharField(max_length=20, default="queued")
    result = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
