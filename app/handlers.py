from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from app.serializers import PredictSerializer
from app.di import prediction_service
from app.tasks import run_inference

class PredictHandler(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PredictSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        prediction = prediction_service.create(serializer.validated_data)
        run_inference.delay(str(prediction.id), serializer.validated_data)
        return Response({"prediction_id": prediction.id}, status=status.HTTP_202_ACCEPTED)
