from django.contrib import admin

from app.models.player import Player


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("player_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("name",)
