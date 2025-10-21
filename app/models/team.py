from django.db import models

class Team(models.Model):
    team_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.team_id})"

    class Meta:
        verbose_name = "Team"
        verbose_name_plural = "Teams"
