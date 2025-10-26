import json

from channels.generic.websocket import AsyncWebsocketConsumer


class PredictionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(
            text_data=json.dumps({"message": "Connected to prediction WebSocket"})
        )

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        await self.send(
            text_data=json.dumps({"message": "Received your data", "data": data})
        )

    async def send_prediction(self, event):
        await self.send(
            text_data=json.dumps(
                {
                    "team1_win_probability": event["team1_win_probability"],
                    "team2_win_probability": event["team2_win_probability"],
                }
            )
        )
