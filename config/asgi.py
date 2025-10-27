import os

import django
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import path

from app.consumers.forecast import make_forecast_consumer

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

forecast_consumer = make_forecast_consumer("ml_forecasts")

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AuthMiddlewareStack(
            URLRouter(
                [
                    path("api/v1/ws/forecast/", forecast_consumer.as_asgi()),
                ]
            )
        ),
    }
)
