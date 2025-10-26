from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer


def send_prediction_to_ws(prediction: dict):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "predictions",
        {
            "type": "send_prediction",
            "team1_win_probability": prediction["team1_win_probability"],
            "team2_win_probability": prediction["team2_win_probability"],
        },
    )
