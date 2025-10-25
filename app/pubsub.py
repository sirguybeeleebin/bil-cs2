import json
from config import settings

def publish_prediction_event(prediction_id: str, event: dict):
    payload = {"prediction_id": prediction_id, "event": event}
    settings.REDIS_CLIENT.publish("predictions", json.dumps(payload))
