from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from extract_from_json_load_to_clickhouse.extract_from_json_load_to_clickhouse import extract_from_json_load_to_clickhouse  

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="extract_from_json_load_to_clickhouse",
    default_args=DEFAULT_ARGS,
    description="ETL JSON games into ClickHouse",
    start_date=datetime.today().replace(hour=0, minute=0, second=0, microsecond=0), 
    schedule_interval="@daily",
    catchup=False,
) as dag:

    run_etl = PythonOperator(
        task_id="extract_from_json_load_to_clickhouse",
        python_callable=extract_from_json_load_to_clickhouse,
        op_kwargs={
            "clickhouse_host": "localhost",
            "clickhouse_port": 9000,
            "clickhouse_user": "cs2_user",
            "clickhouse_password": "cs2_password",
            "clickhouse_db": "cs2_db",
            "drop_table": True,
            "path_to_games_raw_dir": "data/games_raw",
        },
    )

    run_etl
