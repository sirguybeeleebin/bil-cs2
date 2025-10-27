import json

from channels.generic.websocket import AsyncWebsocketConsumer


def make_forecast_consumer(group: str, channel_layer_provider=None):
    class ForecastConsumer(AsyncWebsocketConsumer):
        async def connect(self):
            self._channel_layer = channel_layer_provider or self.channel_layer
            await self._channel_layer.group_add(group, self.channel_name)
            await self.accept()
            await self.send(
                text_data=json.dumps(
                    {"message": "Подключение к WebSocket ml_forecasts установлено"}
                )
            )

        async def disconnect(self, close_code):
            await self._channel_layer.group_discard(group, self.channel_name)

        async def receive(self, text_data=None, bytes_data=None):
            if text_data:
                try:
                    data = json.loads(text_data)
                except json.JSONDecodeError:
                    data = text_data
                await self.send(
                    text_data=json.dumps({"message": "Данные получены", "data": data})
                )

        async def send_ml_forecast(self, event):
            # Channels ожидает, что event["type"] -> имя метода
            await self.send(text_data=json.dumps(event["event"]))

    return ForecastConsumer
