import logging

from celery.result import AsyncResult
from django.contrib.auth.models import User
from rest_framework import generics, permissions, serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView, Request

from backend.di import map_repo, player_repo, team_repo
from backend.tasks import ml_forecast_inference_task

log = logging.getLogger(__name__)


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ("username", "password", "email")

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            password=validated_data["password"],
            email=validated_data.get("email", ""),
        )
        return user


class RegisterHandler(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer


class MapSearchHandler(APIView):
    permission_classes = [permissions.IsAuthenticated]
    repo = map_repo

    def get(self, request: Request) -> Response:
        name = request.query_params.get("name", "")
        limit = (
            int(request.query_params.get("limit", 10))
            if request.query_params.get("limit", "").isdigit()
            else 10
        )
        offset = (
            int(request.query_params.get("offset", 0))
            if request.query_params.get("offset", "").isdigit()
            else 0
        )
        results = self.repo.search_by_name(name=name, limit=limit, offset=offset)
        return Response(results, status=status.HTTP_200_OK)


class TeamSearchHandler(APIView):
    permission_classes = [permissions.IsAuthenticated]
    repo = team_repo

    def get(self, request: Request) -> Response:
        name = request.query_params.get("name", "")
        limit = (
            int(request.query_params.get("limit", 10))
            if request.query_params.get("limit", "").isdigit()
            else 10
        )
        offset = (
            int(request.query_params.get("offset", 0))
            if request.query_params.get("offset", "").isdigit()
            else 0
        )
        results = self.repo.search_by_name(name=name, limit=limit, offset=offset)
        return Response(results, status=status.HTTP_200_OK)


class PlayerSearchHandler(APIView):
    permission_classes = [permissions.IsAuthenticated]
    repo = player_repo

    def get(self, request: Request) -> Response:
        name = request.query_params.get("name", "")
        limit = (
            int(request.query_params.get("limit", 10))
            if request.query_params.get("limit", "").isdigit()
            else 10
        )
        offset = (
            int(request.query_params.get("offset", 0))
            if request.query_params.get("offset", "").isdigit()
            else 0
        )
        results = self.repo.search_by_name(name=name, limit=limit, offset=offset)
        return Response(results, status=status.HTTP_200_OK)


class ForecastRequestSerializer(serializers.Serializer):
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


class ForecastResponseSerializer(serializers.Serializer):
    task_id = serializers.CharField()


class ForecastResultSerializer(serializers.Serializer):
    task_id = serializers.CharField()
    team1_id = serializers.IntegerField()
    team2_id = serializers.IntegerField()
    team1_win_probability = serializers.FloatField()
    team2_win_probability = serializers.FloatField()


class ForecastHandler(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = ForecastRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        log.info(f"Запуск задачи прогнозирования с входными данными: {data}")
        try:
            task = ml_forecast_inference_task.apply_async(args=[data])
            log.info(f"Celery task запущена: {task.id}")
            response_serializer = ForecastResponseSerializer({"task_id": task.id})
            return Response(response_serializer.data, status=status.HTTP_202_ACCEPTED)
        except Exception as e:
            log.exception(f"Ошибка при запуске задачи прогнозирования: {e}")
            return Response(
                {"detail": "Ошибка при запуске задачи"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ForecastResultHandler(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request, task_id: str) -> Response:
        result = AsyncResult(task_id)

        if result.state == "PENDING":
            return Response(
                {"status": "PENDING", "task_id": task_id},
                status=status.HTTP_202_ACCEPTED,
            )
        if result.failed():
            return Response(
                {"status": "FAILED", "task_id": task_id, "error": str(result.result)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if result.successful():
            data = result.result
            data["task_id"] = task_id
            response_serializer = ForecastResultSerializer(data=data)
            if response_serializer.is_valid():
                return Response(response_serializer.data, status=status.HTTP_200_OK)
            return Response(
                response_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(
            {"status": result.status, "task_id": task_id},
            status=status.HTTP_202_ACCEPTED,
        )
