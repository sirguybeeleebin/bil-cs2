from django.urls import path
from app.handlers import PredictHandler

urlpatterns = [
    path("predict/", PredictHandler.as_view()),
]
