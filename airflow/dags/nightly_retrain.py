# airflow/dags/nightly_retrain.py
from __future__ import annotations
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"owner": "airflow", "retries": 2}

with DAG(
    dag_id="nightly_retrain_als",
    description="Merge feedback and retrain ALS nightly",
    default_args=default_args,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule="15 2 * * *",  # 02:15 UTC daily
    catchup=False,
    tags=["ml", "als", "retrain"],
) as dag:

    # 1) (Optional) Merge feedback rows; keep it lightweight for demo
    merge_feedback = BashOperator(
        task_id="merge_feedback",
        bash_command=(
            "python - <<'PY'\n"
            "import pandas as pd, pathlib\n"
            "ratings='data/raw/ratings.csv'\n"
            "fb='data/feedback/reviews.csv'\n"
            "if pathlib.Path(fb).exists():\n"
            "    f=pd.read_csv(fb)\n"
            "    print('Feedback rows:', len(f))\n"
            "else:\n"
            "    print('No feedback file found; skipping merge')\n"
            "PY"
        ),
    )

    # 2) Retrain ALS using your project module (runtime, not at DAG import)
    retrain = BashOperator(
        task_id="train_als",
        bash_command=(
            "python -m src.models.train_als "
            "--ratings_csv data/raw/ratings.csv "
            "--movies_csv  data/raw/movies.csv "
            "--model_dir   models/als_best"
        ),
    )

    merge_feedback >> retrain
