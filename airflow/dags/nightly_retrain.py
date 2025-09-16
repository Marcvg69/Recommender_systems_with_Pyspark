# airflow/dags/nightly_retrain.py
import os
import pathlib
import pendulum
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# ---- Paths & timezone --------------------------------------------------------
TZ = pendulum.timezone("Europe/Brussels")

# Auto-detect repo root if Airflow runs from <repo>/airflow/dags/nightly_retrain.py
# Option B (Airflow in Docker): override via env: REPO_DIR=/opt/airflow/repo
DEFAULT_REPO_DIR = str(pathlib.Path(__file__).resolve().parents[2])  # <repo>
REPO_DIR = os.environ.get("REPO_DIR", DEFAULT_REPO_DIR)

# Data locations relative to REPO_DIR
DEFAULT_RATINGS = "data/raw/ratings.csv"
FEEDBACK = "data/feedback/ratings_feedback.csv"
MERGED = "data/working/ratings_merged.csv"

# Spark memory (can override via env on the Airflow worker if needed)
SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "2g")

# ---- Commands ---------------------------------------------------------------
# Merge base ratings + feedback (if exists) using inline Python (no external deps beyond pandas)
merge_cmd = rf"""
set -euo pipefail
cd "{REPO_DIR}"
python - << 'PY'
import os, pandas as pd, pathlib
pathlib.Path("data/working").mkdir(parents=True, exist_ok=True)
base_path = "{DEFAULT_RATINGS}"
fb_path   = "{FEEDBACK}"
out_path  = "{MERGED}"
base = pd.read_csv(base_path)
if os.path.exists(fb_path):
    fb = pd.read_csv(fb_path)
    df = pd.concat([base[['userId','movieId','rating','timestamp']], fb], ignore_index=True)
else:
    df = base
df.to_csv(out_path, index=False)
print(f"Wrote merged ratings to {{out_path}} (rows={{len(df)}})")
PY
"""

# Train & precompute sims INSIDE your Docker 'app' service (no local Java needed)
train_cmd = rf"""
set -euo pipefail
cd "{REPO_DIR}"
docker compose run --rm \
  -e SPARK_DRIVER_MEMORY={SPARK_DRIVER_MEMORY} \
  app bash -lc '
python -m src.models.tune_als \
  --ratings_csv {MERGED} \
  --model_out models/als_best \
  --ranks 16,32,64 \
  --regParams 0.01,0.05,0.1 \
  --maxIter 15 \
  --sims_topN 30 \
  --sims_method driver \
  --driver_batch 256 \
  --shuffle_partitions 32
'
"""

# ---- DAG --------------------------------------------------------------------
with DAG(
    dag_id="nightly_retrain_als",
    description="Nightly ALS retrain + item-sim precompute (runs training inside Docker).",
    schedule="15 2 * * *",            # 02:15 daily (Europe/Brussels)
    start_date=datetime(2025, 9, 1, tzinfo=TZ),
    catchup=False,
    tags=["recsys", "als", "spark"],
) as dag:

    merge = BashOperator(
        task_id="merge_feedback",
        bash_command=merge_cmd,
    )

    retrain = BashOperator(
        task_id="tune_and_save",
        bash_command=train_cmd,
    )

    merge >> retrain
