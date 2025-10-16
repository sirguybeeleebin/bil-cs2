from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_test_split.train_test_split import train_test_split  

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="train_test_split",
    default_args=DEFAULT_ARGS,
    description="Generate train/test split from ClickHouse games",
    start_date=datetime.now(),  # start from today
    schedule_interval="@daily",
    catchup=False,
) as dag:

    run_split = PythonOperator(
        task_id="train_test_split",
        python_callable=train_test_split,
        op_kwargs={
            "clickhouse_host": "localhost",
            "clickhouse_port": 9000,
            "clickhouse_user": "cs2_user",
            "clickhouse_password": "cs2_password",
            "clickhouse_db": "cs2_db",
            "output_dir": "data/train_test_splits",
        },
    )

    run_split
