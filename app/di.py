from app.repositories import PredictionRepository
from app.services import PredictionService

prediction_repository = PredictionRepository()
prediction_service = PredictionService(prediction_repository)
