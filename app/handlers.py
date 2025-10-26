from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.di import prediction_repository
from app.serializers import PredictionRequestSerializer
from app.tasks import run_inference


class PredictHandler(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        team1_id = data["team1_id"]
        team2_id = data["team2_id"]

        prediction_data = {
            "map_id": data["map_id"],
            "team1_id": team1_id,
            "team2_id": team2_id,
            "start_ct_team_id": data["start_ct_team_id"],
            "team1_player1_id": data["team1_player1_id"],
            "team1_player2_id": data["team1_player2_id"],
            "team1_player3_id": data["team1_player3_id"],
            "team1_player4_id": data["team1_player4_id"],
            "team1_player5_id": data["team1_player5_id"],
            "team2_player1_id": data["team2_player1_id"],
            "team2_player2_id": data["team2_player2_id"],
            "team2_player3_id": data["team2_player3_id"],
            "team2_player4_id": data["team2_player4_id"],
            "team2_player5_id": data["team2_player5_id"],
        }
        prediction = prediction_repository.upsert(prediction_data)

        run_inference.delay(str(prediction["prediction_id"]))

        return Response(
            {
                "prediction_id": prediction["prediction_id"],
                "team1_id": team1_id,
                "team2_id": team2_id,
                "status": "inference_started",
            }
        )
