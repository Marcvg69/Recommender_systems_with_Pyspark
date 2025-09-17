# airflow/dags/nightly_retrain.py
from __future__ import annotations

import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "retries": 2,
}

with DAG(
    dag_id="nightly_retrain_als",
    description="Merge feedback and retrain ALS nightly",
    default_args=default_args,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="15 2 * * *",  # 02:15
    catchup=False,
    tags=["ml", "als", "retrain"],
) as dag:

    merge_feedback = BashOperator(
        task_id="merge_feedback",
        bash_command="""
set -euo pipefail
cd "$REPO_DIR"

mkdir -p data/working

# start from the canonical ratings.csv
cp -f data/raw/ratings.csv data/working/ratings_merged.csv

# append any feedback CSVs if present (skip their headers)
shopt -s nullglob
if compgen -G "data/feedback/*.csv" > /dev/null; then
  for f in data/feedback/*.csv; do
    tail -n +2 "$f" >> data/working/ratings_merged.csv || true
  done
fi
""",
    )

    tune_and_save = BashOperator(
        task_id="tune_and_save",
        bash_command="""
set -euo pipefail
cd "$REPO_DIR"

# Run the app service from the *host* repo so bind-mounts resolve on macOS Docker
docker compose \
  --project-directory "$REPO_DIR_HOST" \
  -f "$REPO_DIR_HOST/docker-compose.yml" \
  run --rm -e SPARK_DRIVER_MEMORY=2g app bash -lc '
python -m src.models.tune_als \
  --ratings_csv data/working/ratings_merged.csv \
  --model_out models/als_best \
  --ranks 16,32,64 \
  --regParams 0.01,0.05,0.1 \
  --maxIter 15 \
  --sims_topN 30 \
  --sims_method driver \
  --driver_batch 256 \
  --shuffle_partitions 32
'
""",
    )

    merge_feedback >> tune_and_save


