import json
import uuid
from datetime import datetime, time
from pathlib import Path

import joblib
import mlflow
import redis
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from ml.ml_pipeline import run_ml_pipeline

# -----------------------------
# Config
# -----------------------------
default_args = {"owner": "airflow", "depends_on_past": False, "retries": 1}

DATA_PATH = Variable.get("DATA_PATH", default_var="data/games_raw")
ML_RESULTS_PATH = Variable.get("ML_RESULTS_PATH", default_var="data/ml_results")
TEST_SIZE = int(Variable.get("TEST_SIZE", default_var=100))
N_SPLITS = int(Variable.get("N_SPLITS", default_var=10))
N_ITERS = int(Variable.get("N_ITERS", default_var=10))
RANDOM_STATE = int(Variable.get("RANDOM_STATE", default_var=42))
MLFLOW_TRACKING_URI = Variable.get(
    "MLFLOW_TRACKING_URI", default_var=f"file://{Path(ML_RESULTS_PATH) / 'mlflow'}"
)
REDIS_HOST = Variable.get("REDIS_HOST", default_var="localhost")
REDIS_PORT = int(Variable.get("REDIS_PORT", default_var=6379))
REDIS_CHANNEL = Variable.get("REDIS_CHANNEL", default_var="ml_updates")

Path(ML_RESULTS_PATH).mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ml_stacking_pipeline")

start_date = datetime.combine(datetime.today(), time.min)


# -----------------------------
# Tasks
# -----------------------------
def task_train_pipeline(**kwargs):
    pipeline, metrics = run_ml_pipeline(
        data_path=DATA_PATH,
        test_size=TEST_SIZE,
        n_splits=N_SPLITS,
        n_iters=N_ITERS,
        random_state=RANDOM_STATE,
    )
    # сохраняем в XCom
    kwargs["ti"].xcom_push(key="pipeline", value=pipeline)
    kwargs["ti"].xcom_push(key="metrics", value=metrics)
    kwargs["ti"].xcom_push(key="pipeline_uuid", value=str(uuid.uuid4()))


def task_save_pipeline(**kwargs):
    ti = kwargs["ti"]
    pipeline = ti.xcom_pull(key="pipeline")
    metrics = ti.xcom_pull(key="metrics")
    pipeline_uuid = ti.xcom_pull(key="pipeline_uuid")

    metrics_path = Path(ML_RESULTS_PATH) / f"{pipeline_uuid}.json"
    pipeline_path = Path(ML_RESULTS_PATH) / f"{pipeline_uuid}.joblib"

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)
    joblib.dump(pipeline, pipeline_path)

    ti.xcom_push(key="metrics_path", value=str(metrics_path))
    ti.xcom_push(key="pipeline_path", value=str(pipeline_path))


def task_log_mlflow(**kwargs):
    ti = kwargs["ti"]
    pipeline_path = ti.xcom_pull(key="pipeline_path")
    metrics_path = ti.xcom_pull(key="metrics_path")
    metrics = ti.xcom_pull(key="metrics")
    pipeline_uuid = ti.xcom_pull(key="pipeline_uuid")

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

        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")


def task_publish_redis(**kwargs):
    ti = kwargs["ti"]
    pipeline_uuid = ti.xcom_pull(key="pipeline_uuid")
    pipeline_path = ti.xcom_pull(key="pipeline_path")
    metrics_path = ti.xcom_pull(key="metrics_path")

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    message_data = {
        "pipeline_id": pipeline_uuid,
        "metrics_file": metrics_path,
        "model_file": pipeline_path,
    }
    redis_client.publish(REDIS_CHANNEL, json.dumps(message_data))


# -----------------------------
# DAG
# -----------------------------
with DAG(
    dag_id="ml_pipeline_steps",
    default_args=default_args,
    start_date=start_date,
    schedule_interval="0 0 * * *",
    catchup=False,
    tags=["ml", "stacking"],
) as dag:
    train_pipeline = PythonOperator(
        task_id="train_pipeline",
        python_callable=task_train_pipeline,
        provide_context=True,
    )

    save_pipeline = PythonOperator(
        task_id="save_pipeline",
        python_callable=task_save_pipeline,
        provide_context=True,
    )

    log_mlflow = PythonOperator(
        task_id="log_mlflow",
        python_callable=task_log_mlflow,
        provide_context=True,
    )

    publish_redis = PythonOperator(
        task_id="publish_redis",
        python_callable=task_publish_redis,
        provide_context=True,
    )

    # -----------------------------
    # Dependencies
    # -----------------------------
    train_pipeline >> save_pipeline >> log_mlflow >> publish_redis
