from django.urls import path, re_path

from app import consumers
from app.handlers import PredictHandler

urlpatterns = [
    path("predict/", PredictHandler.as_view(), name="predict"),
    re_path(r"ws/predictions/$", consumers.PredictionConsumer.as_asgi()),
]
