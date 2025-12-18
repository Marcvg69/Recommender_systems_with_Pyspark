# airflow/dags/nightly_retrain.py
from __future__ import annotations
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

# Adjust this if your repo lives in a subfolder under /opt/airflow/dags
PROJECT_ROOT = "/opt/airflow/dags"

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

    # 1) (Optional) Log feedback rows (kept lightweight)
    merge_feedback = BashOperator(
        task_id="merge_feedback",
        bash_command=(
            "set -euo pipefail\n"
            f"cd {PROJECT_ROOT}\n"
            "export PYTHONPATH=$(pwd):$PYTHONPATH\n"
            "python - <<'PY'\n"
            "import pandas as pd, pathlib\n"
            "fb = pathlib.Path('data/feedback/reviews.csv')\n"
            "print('CWD:', pathlib.Path.cwd())\n"
            "if fb.exists():\n"
            "    f = pd.read_csv(fb)\n"
            "    print('Feedback rows:', len(f))\n"
            "else:\n"
            "    print('No feedback file found; skipping merge')\n"
            "PY\n"
        ),
    )

    # 2) Retrain ALS (robust: set cwd, PYTHONPATH, ensure output dir)
    retrain = BashOperator(
        task_id="train_als",
        bash_command=(
            "set -euo pipefail\n"
            f"cd {PROJECT_ROOT}\n"
            "export PYTHONPATH=$(pwd):$PYTHONPATH\n"
            "mkdir -p models/als_best\n"
            "python -m src.models.train_als "
            "--ratings_csv data/raw/ratings.csv "
            "--movies_csv  data/raw/movies.csv "
            "--model_dir   models/als_best\n"
        ),
    )

    merge_feedback >> retrain
