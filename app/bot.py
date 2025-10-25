import json
from config import settings

def listen_to_predictions():
    pubsub = settings.REDIS_CLIENT.pubsub()
    pubsub.subscribe("predictions")
    print("Telegram bot listening...")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue
        data = json.loads(message["data"])
        prediction_id = data["prediction_id"]
        event = data["event"]

        if "progress" in event:
            text = f"Prediction {prediction_id} progress: {event['progress']}%"
        elif "status" in event and event["status"] == "done":
            text = f"Prediction {prediction_id} done!\nResult: {event['result']}"
        else:
            text = f"Prediction {prediction_id} update: {event}"

        settings.TELEGRAM_BOT.send_message(chat_id=settings.TELEGRAM_CHAT_ID, text=text)

if __name__ == "__main__":
    listen_to_predictions()
