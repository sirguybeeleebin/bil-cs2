from django.db import models

class Player(models.Model):
    player_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.player_id})"

    class Meta:
        verbose_name = "Player"
        verbose_name_plural = "Players"
