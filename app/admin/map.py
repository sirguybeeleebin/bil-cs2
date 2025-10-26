from django.contrib import admin

from app.models.map import Map


@admin.register(Map)
class MapAdmin(admin.ModelAdmin):
    list_display = ("map_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("created_at",)
