import json
import uuid
from datetime import datetime, time
from pathlib import Path

import joblib
import mlflow
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from ml.ml_pipeline import run_ml_pipeline

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

DATA_PATH = Variable.get("DATA_PATH", default_var="data/games_raw")
ML_RESULTS_PATH = Variable.get("ML_RESULTS_PATH", default_var="data/ml_results")
TEST_SIZE = int(Variable.get("TEST_SIZE", default_var=100))
N_SPLITS = int(Variable.get("N_SPLITS", default_var=10))
N_ITERS = int(Variable.get("N_ITERS", default_var=10))
RANDOM_STATE = int(Variable.get("RANDOM_STATE", default_var=42))
MLFLOW_TRACKING_URI = Variable.get(
    "MLFLOW_TRACKING_URI", default_var=f"file://{Path(ML_RESULTS_PATH) / 'mlflow'}"
)

Path(ML_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ml_stacking_pipeline")

start_date = datetime.combine(datetime.today(), time.min)


def execute_run_ml_pipeline(**kwargs):
    pipeline, metrics = run_ml_pipeline(
        data_path=DATA_PATH,
        test_size=TEST_SIZE,
        n_splits=N_SPLITS,
        n_iters=N_ITERS,
        random_state=RANDOM_STATE,
    )

    pipeline_uuid = str(uuid.uuid4())
    metrics_path = Path(ML_RESULTS_PATH) / f"{pipeline_uuid}.json"
    pipeline_path = Path(ML_RESULTS_PATH) / f"{pipeline_uuid}.joblib"

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)
    joblib.dump(pipeline, pipeline_path)

    with mlflow.start_run(run_name=pipeline_uuid):
        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "n_splits": N_SPLITS,
                "n_iters": N_ITERS,
                "random_state": RANDOM_STATE,
            }
        )
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(pipeline_path), artifact_path="pipeline")

    print("✅ Pipeline executed successfully!")
    print("Model:", pipeline_path)
    print("Metrics:", metrics_path)


with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    start_date=start_date,
    schedule_interval="0 0 * * *",
    catchup=False,
    tags=["ml", "stacking"],
) as dag:
    run_pipeline_task = PythonOperator(
        task_id="run_ml_pipeline_task",
        python_callable=execute_run_ml_pipeline,
    )
