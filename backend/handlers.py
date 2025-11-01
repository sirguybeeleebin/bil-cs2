from django.contrib.auth.models import User
from django.db import IntegrityError
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.di import dictionary_service, forecaster_service
from backend.models import TrainMetric, TrainResult


class MapResponse(serializers.Serializer):
    map_id = serializers.IntegerField()
    name = serializers.CharField()


class TeamResponse(serializers.Serializer):
    team_id = serializers.IntegerField()
    name = serializers.CharField()


class PlayerResponse(serializers.Serializer):
    player_id = serializers.IntegerField()
    name = serializers.CharField()


class ForecastRequest(serializers.Serializer):
    map_id = serializers.IntegerField()
    team1_id = serializers.IntegerField()
    team2_id = serializers.IntegerField()
    team1_player1_id = serializers.IntegerField()
    team1_player2_id = serializers.IntegerField()
    team1_player3_id = serializers.IntegerField()
    team1_player4_id = serializers.IntegerField()
    team1_player5_id = serializers.IntegerField()
    team2_player1_id = serializers.IntegerField()
    team2_player2_id = serializers.IntegerField()
    team2_player3_id = serializers.IntegerField()
    team2_player4_id = serializers.IntegerField()
    team2_player5_id = serializers.IntegerField()


class ForecastResponse(serializers.Serializer):
    forecast_id = serializers.CharField()


class ForecastResultResponse(serializers.Serializer):
    forecast_id = serializers.CharField()
    team1_id = serializers.IntegerField()
    team2_id = serializers.IntegerField()
    team1_win_probability = serializers.FloatField()
    team2_win_probability = serializers.FloatField()


class RegisterRequest(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(max_length=128, write_only=True)


class RegisterResponse(serializers.Serializer):
    user_id = serializers.CharField()


class RegisterHandler(APIView):
    def post(self, request):
        serializer = RegisterRequest(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            user = User.objects.create_user(
                username=data["username"], password=data["password"]
            )
        except IntegrityError:
            return Response(
                {"detail": "Username already exists."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        response_serializer = RegisterResponse({"user_id": user.id})
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


class MapHandler(APIView):
    def get(self, request):
        name = request.GET.get("name", "")
        page = int(request.GET.get("page", 1))
        page_size = int(request.GET.get("page_size", 10))
        results = dictionary_service.search_map_by_name(
            name=name, page=page, per_page=page_size
        )
        serializer = MapResponse(results, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class TeamHandler(APIView):
    def get(self, request):
        name = request.GET.get("name", "")
        page = int(request.GET.get("page", 1))
        page_size = int(request.GET.get("page_size", 10))
        results = dictionary_service.search_team_by_name(
            name=name, page=page, per_page=page_size
        )
        serializer = TeamResponse(results, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class PlayerHandler(APIView):
    def get(self, request):
        name = request.GET.get("name", "")
        page = int(request.GET.get("page", 1))
        page_size = int(request.GET.get("page_size", 10))
        results = dictionary_service.search_player_by_name(
            name=name, page=page, per_page=page_size
        )
        serializer = PlayerResponse(results, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ForecastHandler(APIView):
    def post(self, request):
        serializer = ForecastRequest(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        forecast_id = forecaster_service.forecast(
            map_id=data["map_id"],
            team1_id=data["team1_id"],
            team2_id=data["team2_id"],
            team1_player1_id=data["team1_player1_id"],
            team1_player2_id=data["team1_player2_id"],
            team1_player3_id=data["team1_player3_id"],
            team1_player4_id=data["team1_player4_id"],
            team1_player5_id=data["team1_player5_id"],
            team2_player1_id=data["team2_player1_id"],
            team2_player2_id=data["team2_player2_id"],
            team2_player3_id=data["team2_player3_id"],
            team2_player4_id=data["team2_player4_id"],
            team2_player5_id=data["team2_player5_id"],
        )

        response_serializer = ForecastResponse({"forecast_id": forecast_id})
        return Response(response_serializer.data, status=status.HTTP_202_ACCEPTED)


class ForecastResultHandler(APIView):
    def get(self, request, forecast_id):
        task_result = forecaster_service.get_forecast_result_by_id(forecast_id)
        state = task_result.state

        if state == "PENDING":
            return Response(
                {"status": "PENDING", "message": "Task is still running."},
                status=status.HTTP_200_OK,
            )
        elif state == "FAILURE":
            return Response(
                {"status": "FAILURE", "message": "Task execution failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        elif state == "SUCCESS":
            result_data = task_result.result
            result_data["forecast_id"] = forecast_id
            serializer = ForecastResultResponse(data=result_data)
            if serializer.is_valid():
                return Response(
                    {"status": "SUCCESS", "result": serializer.data},
                    status=status.HTTP_200_OK,
                )
            else:
                return Response(
                    {"status": "FAILURE", "message": "Invalid result format"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
            return Response({"status": state}, status=status.HTTP_200_OK)


class MLMetricsResponse(serializers.Serializer):
    train_result_id = serializers.UUIDField()
    auc = serializers.FloatField(allow_null=True)
    f1 = serializers.FloatField(allow_null=True)
    precision = serializers.FloatField(allow_null=True)
    recall = serializers.FloatField(allow_null=True)
    accuracy = serializers.FloatField(allow_null=True)
    tp = serializers.IntegerField(allow_null=True)
    tn = serializers.IntegerField(allow_null=True)
    fp = serializers.IntegerField(allow_null=True)
    fn = serializers.IntegerField(allow_null=True)
    created_at = serializers.DateTimeField()


class MLMetricsHandler(APIView):
    def get(self, request):
        latest_train_result = TrainResult.objects.order_by("-created_at").first()
        if not latest_train_result:
            return Response(
                {"detail": "No training results found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get the associated TrainMetric
        try:
            train_metric = latest_train_result.train_metric
        except TrainMetric.DoesNotExist:
            return Response(
                {"detail": "Metrics for the latest training result not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = MLMetricsResponse(
            {
                "train_result_id": latest_train_result.train_result_id,
                "auc": train_metric.auc,
                "f1": train_metric.f1,
                "precision": train_metric.precision,
                "recall": train_metric.recall,
                "accuracy": train_metric.accuracy,
                "tp": train_metric.tp,
                "tn": train_metric.tn,
                "fp": train_metric.fp,
                "fn": train_metric.fn,
                "created_at": train_metric.created_at,
            }
        )
        return Response(serializer.data, status=status.HTTP_200_OK)
