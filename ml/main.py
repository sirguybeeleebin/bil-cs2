import warnings
import argparse
import os
import hashlib
import json
import pickle
import logging
from datetime import datetime

import pika
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from dotenv import load_dotenv
from ml.metrics import get_metrics

from ml.feature_extraction import (
    ColumnSelector,
    PlayerBagEncoder,    
    TeamBagEncoder, 
    PlayerEloEncoder,    
    PlayerMapEloEncoder,   
)
from ml.feature_selection import select_features_with_logit_and_cv
from ml.load_data import get_game_ids, get_X_y

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def get_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_file", type=str, default=".env", help="Path to .env file"
    )
    args, _ = parser.parse_known_args()
    load_dotenv(args.env_file)
    
    settings = {
        "PATH_TO_GAMES_RAW": os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw"),
        "PATH_TO_ML_RESULTS": os.getenv("PATH_TO_ML_RESULTS", "data/ml"),
        "RABBITMQ_URL": os.getenv("RABBITMQ_URL", "amqp://cs_user:cs_password@localhost:5672/"),
        "RABBITMQ_EXCHANGE_NAME": os.getenv("RABBITMQ_EXCHANGE_NAME", "cs_exchange"),
        "RABBITMQ_EXCHANGE_TYPE": os.getenv("RABBITMQ_EXCHANGE_TYPE", "direct"),  
        "RABBITMQ_ROUTING_KEY": os.getenv("RABBITMQ_ROUTING_KEY", "cs_routing_key")
    }
    return settings


def run(
    path_to_game_raw_dir: str = "data/games_raw",
    path_to_results_dir: str = "data/ml",    
):
    TEST_SIZE = 100
    C_GRID = np.linspace(0.00001, 0.01, 100)
    SCORING = "roc_auc"
    VERBOSE = 2
    N_SPLITS = 10
    TSCV = TimeSeriesSplit(N_SPLITS)

    log.info("Загрузка ID игр...")
    game_ids = get_game_ids(path_to_game_raw_dir)
    game_ids_train, game_ids_test = game_ids[:-TEST_SIZE], game_ids[-TEST_SIZE:]
    log.info(f"Всего игр: {len(game_ids)}, обучение: {len(game_ids_train)}, тест: {len(game_ids_test)}")

    X_train, y_train = get_X_y(path_to_game_raw_dir, game_ids_train)
    X_test, y_test = get_X_y(path_to_game_raw_dir, game_ids_test) 
    
    MAP_ID_COL = [0]
    ONE_HOT_COLS = [0, 1]
    TEAM_COLS = [2, 3]
    PLAYER_COLS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    log.info("Построение пайплайна признаков...")
    feature_pipeline = Pipeline(
        [
            (
                "encoder",
                FeatureUnion(
                    [
                        ("one_hot", Pipeline([
                            ("select_cols", ColumnSelector(ONE_HOT_COLS)),
                            ("ohe", OneHotEncoder(handle_unknown="ignore"))
                        ])),
                        ("team_bag", Pipeline([
                            ("select_teams", ColumnSelector(TEAM_COLS)),
                            ("team_encoder", TeamBagEncoder())
                        ])),
                        ("player_bag", Pipeline([
                            ("select_players", ColumnSelector(PLAYER_COLS)),
                            ("player_encoder", PlayerBagEncoder())
                        ])),
                        ("player_elo_1", Pipeline([
                            ("select_players", ColumnSelector(PLAYER_COLS)),
                            ("elo_encoder", PlayerEloEncoder()),
                            ("scaler", MinMaxScaler())
                        ])),  
                        ("player_elo_2", Pipeline([
                            ("select_players", ColumnSelector(MAP_ID_COL + PLAYER_COLS)),
                            ("elo_encoder", PlayerMapEloEncoder()),
                            ("scaler", MinMaxScaler())
                        ])),                              
                    ]
                ),
            ),           
        ]
    )

    log.info("Извлечение признаков...")
    feature_pipeline.fit(X_train, y_train)
    
    X_train = feature_pipeline.transform(X_train)
    X_test = feature_pipeline.transform(X_test)
    
    log.info("Отбор признаков...")
    mask = select_features_with_logit_and_cv(
        X_train,
        y_train,
        Cs=C_GRID,
        scoring=SCORING,
        cv=TSCV, 
        verbose=VERBOSE,   
    )
    
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]    
   
    param_grid = {"C": np.linspace(0.00001, 1, 1000)}
    grid_search_final = GridSearchCV(
        estimator=LogisticRegression(solver="liblinear"),
        param_grid=param_grid,
        cv=TSCV,
        scoring=SCORING,
        n_jobs=-1,
        verbose=VERBOSE
    )
    grid_search_final.fit(X_train, y_train)
    final_clf = grid_search_final.best_estimator_
    
    log.info(f"Лучший C для финальной логистической регрессии: {grid_search_final.best_params_['C']}")
    log.info(f"CV ROC AUC: {grid_search_final.best_score_:.4f}")

    y_test_proba = final_clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    metrics = get_metrics(y_test, y_test_pred, y_test_proba)
    log.info(f"Метрики на тесте: {metrics}")

    os.makedirs(path_to_results_dir, exist_ok=True)
    game_ids_bytes = ",".join(map(str, game_ids)).encode()
    hash_id = hashlib.md5(game_ids_bytes).hexdigest()
    created_at = datetime.now().isoformat()    

    metrics_path = os.path.join(path_to_results_dir, f"{hash_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics | {"created_at": created_at}, f, indent=4)

    pipeline_path = os.path.join(path_to_results_dir, f"{hash_id}.pickle")
    with open(pipeline_path, "wb") as f:
        pickle.dump(
            {
                "feature_pipeline": feature_pipeline,
                "feature_selection_mask": mask,
                "best_model": final_clf,
                "created_at": created_at,
            },
            f,
        )

    log.info(json.dumps(metrics, indent=4))
    log.info("Результаты успешно сохранены.")
    
    return pipeline_path, metrics_path


def publish_to_rabbitmq(pipeline_file, metrics_file, rabbitmq_url, exchange_name, exchange_type, routing_key):
    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
    channel = connection.channel()
    
    channel.exchange_declare(exchange=exchange_name, exchange_type=exchange_type, durable=True)

    message = json.dumps({
        "pipeline_file": pipeline_file,
        "metrics_file": metrics_file
    }).encode()

    channel.basic_publish(
        exchange=exchange_name,
        routing_key=routing_key,
        body=message,
        properties=pika.BasicProperties(delivery_mode=2),
    )

    connection.close()
    log.info(f"Published message to exchange '{exchange_name}' (type '{exchange_type}') with routing key '{routing_key}'")


def main():
    settings = get_settings()
    pipeline_path, metrics_path = run(
        path_to_game_raw_dir=settings["PATH_TO_GAMES_RAW"],
        path_to_results_dir=settings["PATH_TO_ML_RESULTS"],
    )
    publish_to_rabbitmq(
        pipeline_file=pipeline_path,
        metrics_file=metrics_path,
        rabbitmq_url=settings["RABBITMQ_URL"],
        exchange_name=settings["RABBITMQ_EXCHANGE_NAME"],
        exchange_type=settings["RABBITMQ_EXCHANGE_TYPE"],
        routing_key=settings["RABBITMQ_ROUTING_KEY"]
    )


if __name__ == "__main__":
    main()
