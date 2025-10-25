from app.models import Prediction

class PredictionRepository:
    def create(self, payload: dict):
        return Prediction.objects.create(payload=payload)

    def update_status(self, prediction_id, status: str):
        Prediction.objects.filter(id=prediction_id).update(status=status)

    def update_result(self, prediction_id, result: dict):
        Prediction.objects.filter(id=prediction_id).update(status="done", result=result)

    def get(self, prediction_id):
        return Prediction.objects.get(id=prediction_id)
