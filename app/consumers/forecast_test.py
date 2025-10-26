import pytest
from channels.testing import WebsocketCommunicator
from django.test import override_settings

from app.consumers.forecast import make_forecast_consumer


@pytest.mark.asyncio
@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
async def test_send_ml_forecast():
    group = "test_group"

    from channels.layers import get_channel_layer

    channel_layer = get_channel_layer()

    consumer_class = make_forecast_consumer(group, channel_layer_provider=channel_layer)
    app = consumer_class.as_asgi()
    communicator = WebsocketCommunicator(app, "/ws/test/")

    connected, _ = await communicator.connect()
    assert connected

    # Пропускаем приветственное сообщение
    welcome = await communicator.receive_json_from()
    assert welcome["message"] == "Подключение к WebSocket ml_forecasts установлено"

    # Отправляем событие через группу
    event = {"prediction": 0.7}
    await channel_layer.group_send(group, {"type": "send_ml_forecast", "event": event})

    # Проверяем, что клиент получил событие
    response = await communicator.receive_json_from()
    assert response == event

    await communicator.disconnect()
