import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from app.consumers import PredictionConsumer
from django.urls import re_path
from app.middleware import JWTAuthMiddlewareHeader

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

websocket_urlpatterns = [
    re_path(r"ws/prediction/(?P<prediction_id>[0-9a-f-]+)/$", PredictionConsumer.as_view()),
]

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": JWTAuthMiddlewareHeader(
        URLRouter(websocket_urlpatterns)
    ),
})
