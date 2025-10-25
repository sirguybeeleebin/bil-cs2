from celery import shared_task
import time
from app.di import prediction_service
from app.pubsub import publish_prediction_event

@shared_task
def run_inference(prediction_id: str, payload: dict):
    prediction_service.set_status(prediction_id, "processing")
    for stage in [10, 40, 70]:
        publish_prediction_event(prediction_id, {"progress": stage})
        time.sleep(1)
    result = {"prediction": 0.82}
    prediction_service.save_result(prediction_id, result)
    publish_prediction_event(prediction_id, {"status": "done", "result": result})
