from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

app = FastAPI(title="CS:GO Winner Prediction API")

client = MlflowClient()

MODEL_NAME = "ml_stacking_pipeline"
MODEL_URI_LATEST = f"models:/{MODEL_NAME}/latest"


class PredictionRequest(BaseModel):
    model_id: str | None = None  # MLflow run_id или registered model version
    features: list[list[float]]  # Input features


@app.get("/")
def root():
    return {
        "status": "running",
        "tracking_uri": mlflow.get_tracking_uri(),
        "default_model": MODEL_URI_LATEST,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        if request.model_id:
            # Загружаем модель по run_id (если задан)
            model_uri = f"runs:/{request.model_id}/pipeline/model"
        else:
            # Используем latest зарегистрированную модель
            model_uri = MODEL_URI_LATEST

        model = mlflow.pyfunc.load_model(model_uri)
        predictions = model.predict(request.features)

        return {
            "model_uri": model_uri,
            "predictions": predictions.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
