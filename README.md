# Recommender Systems with PySpark (ALS + Cold-Start Fallback)

An end-to-end movie recommender:
- **Known users:** PySpark **ALS** model saved to `models/als_best`
- **New users (cold-start):** **item–item cosine** over a sparse items×users matrix, with optional **MMR** diversity
- **UI:** Streamlit
- **Ops:** Airflow DAG for nightly retraining
- **Containers:** Docker + docker-compose

---

## Project Structure

```
.
├─ app/
│  └─ streamlit_app.py             # UI (known user ALS + cold-start + feedback)
├─ src/
│  ├─ __init__.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ train_als.py              # train & save ALS
│  │  ├─ tune_als.py               # grid search / CV
│  │  └─ similarity.py             # CSR build, cosine, popularity baselines, MMR
│  └─ utils/ ...                   # (optional helpers)
├─ airflow/
│  └─ dags/
│     └─ nightly_retrain.py        # import-light DAG (BashOperators)
├─ data/
│  ├─ raw/                         # ratings.csv + movies.csv (small dataset)
│  ├─ sample/                      # tiny sample used in docs
│  └─ feedback/reviews.csv         # user feedback from UI (optional)
├─ models/
│  └─ als_best/                    # saved ALS model (Spark ML format)
├─ Dockerfile
├─ Dockerfile.airflow
├─ docker-compose.yml              # app
├─ docker-compose.airflow.yml      # Airflow (webserver + scheduler)
├─ requirements.txt
└─ README.md
```

---

## Quickstart (Streamlit App)

### Option A: Docker (recommended)

```bash
# 1) build & run the app
docker compose up -d --build app

# 2) open the UI
open http://localhost:8501
```

The app reads defaults from environment variables (set in `docker-compose.yml`):

```yaml
environment:
  - PYTHONPATH=/app
  - RATINGS_CSV=data/raw/ratings.csv
  - MOVIES_CSV=data/raw/movies.csv
  - ALS_MODEL_DIR=models/als_best
```

> **Note:** The Movies CSV and the ALS model must come from the **same dataset**.  
> If titles show as `None`, you’re probably mixing `sample` CSVs with a model trained on `raw` or vice-versa.

### Option B: Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

---

## Features

- **ALS recommendations** for known users (via `recommendForUserSubset`)
- **Cold-start** for new users (select seeds, cosine similarity over a CSR matrix)
- **Popularity baselines** (mean rating × support; IMDB-style weighted)
- **Diversity** (optional MMR re-ranking)
- **Feedback capture** → `data/feedback/reviews.csv` (used by retraining DAG)
- **Caching**: CSR matrix and Spark session/model lazy-loaded and cached

---

## Train & Tune (Spark)

Train on the small dataset already in `data/raw/` and overwrite `models/als_best`:

```bash
# inside the app container
docker compose exec -T app bash -lc "
python -m src.models.train_als   --ratings_csv data/raw/ratings.csv   --movies_csv  data/raw/movies.csv   --model_dir   models/als_best
"
```

Grid search example:

```bash
docker compose exec -T app bash -lc "
python -m src.models.tune_als   --ratings_csv data/raw/ratings.csv   --movies_csv  data/raw/movies.csv   --model_dir   models/als_best
"
```

---

## Airflow (local dev)

We keep Airflow in a separate compose file.

```bash
# validate services exist
docker compose -f docker-compose.airflow.yml config --services
# start
docker compose -f docker-compose.airflow.yml up -d --build airflow-webserver airflow-scheduler
# open UI
open http://localhost:8080
```

**DAG:** `airflow/dags/nightly_retrain.py` is import-light:
- `merge_feedback` (optional)
- `train_als`: `python -m src.models.train_als --ratings_csv data/raw/ratings.csv --movies_csv data/raw/movies.csv --model_dir models/als_best`

**Trigger without CSRF:**
```bash
docker compose -f docker-compose.airflow.yml exec -T airflow-scheduler airflow dags trigger nightly_retrain_als
```

**Compose details:**  
`docker-compose.airflow.yml` mounts your repo and sets `PYTHONPATH` so imports like `from src.models...` work.

```yaml
# snippet
environment:
  AIRFLOW__CORE__LOAD_EXAMPLES: "False"
  AIRFLOW__CORE__EXECUTOR: "SequentialExecutor"
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: "sqlite:////opt/airflow/airflow.db"
  PYTHONPATH: "/opt/airflow:/opt/airflow/repo:/opt/airflow/repo/src"
volumes:
  - ./airflow/dags:/opt/airflow/dags
  - .:/opt/airflow/repo
```

---

## Cold-Start Details (similarity)

We build a sparse **items×users** CSR from ratings, then compute cosine similarity on demand.  
Mitigations for “all results look the same”:

- **Shrinkage**: multiply cosine by `n_overlap / (n_overlap + 100)`  
- **Down-weight heavy raters** (optional TF-IDF/BM25)  
- **MMR re-rank** with λ∈[0.6,0.8] for diversity

All implemented in `src/models/similarity.py`.

---

## Environment Variables

| Variable        | Default                          | Description                                |
|----------------|----------------------------------|--------------------------------------------|
| `RATINGS_CSV`  | `data/raw/ratings.csv`           | Ratings file                               |
| `MOVIES_CSV`   | `data/raw/movies.csv`            | Movies file                                |
| `ALS_MODEL_DIR`| `models/als_best`                | Saved Spark ML model dir                   |
| `PYTHONPATH`   | `/app` or Airflow value shown     | Makes `src.*` importable                   |

---

## Troubleshooting

- **`localhost:8501` refuses** → old container still running, or Streamlit not started on `0.0.0.0`.  
  `docker compose down --remove-orphans && docker compose up -d --build app`

- **Titles show `None`** → dataset mismatch. Ensure Movies CSV **matches** the dataset used to train ALS.

- **`ModuleNotFoundError: src.models.similarity`** → add `src/__init__.py` and `src/models/__init__.py`; set `PYTHONPATH`.

- **Airflow “Broken DAG”** → DAG imported project code at parse time. Use the provided import-light DAG that calls `python -m ...` at runtime.

- **Airflow CSRF “session token missing”** → use the Airflow UI button after logging in, or trigger via CLI as shown above.

---

## Roadmap

- MLflow experiment tracking  
- Better offline ranking metrics (MAP, NDCG, coverage, novelty)  
- User-controllable diversity slider in the UI  
- Model/feature store for production pipelines

---

## Q&A
1) “Why do recommendations across modes look similar?”
Short: small datasets push both ALS and cosine toward popularity hubs.
What we do: raise ALS rank, add cosine shrinkage (down-weight tiny overlaps), and MMR diversity re-rank. Also allow an exclude-popular toggle.
One-liner: “It’s popularity gravity; we counter with rank/shrinkage/MMR and the lists spread out.”
 
2) “How do you handle brand-new users and brand-new items?”
Users: seed with a few liked titles → item–item cosine produces a relevant slate; if no seeds, show popularity.
Items: with zero ratings, ALS can’t place them; we can bootstrap with content (genres/tags) or metadata embeddings, then let ratings take over.
Roadmap: plug in a lightweight content encoder for true item cold-start.
 
3) “How do you evaluate recommendation quality beyond RMSE?”
Offline ranking: Recall@K / Precision@K / NDCG / MAP, plus coverage and novelty.
Ablations: compare ALS only vs ALS+MMR, and cosine with/without shrinkage.
Online: A/B test CTR or add-to-watchlist.
Note: RMSE ≠ ranking quality; that’s why we track ranking metrics too.
 
4) “What does scaling this look like in prod?”
Compute: Spark on a managed cluster (Databricks/EMR), checkpointed training, model registry.
Serving: batch top-N to a feature store/cache, or on-the-fly with a small ALS rank and pre-broadcasted factors.
Ops: Airflow (or orchestration of choice), Docker images per job, observability (latency, failure rate, model freshness, drift alarms).
 
5) “Can you explain why a movie was recommended?”
Cosine path: show “Because you liked X,” and expose top contributing seeds + similarity score.
ALS path: show nearest-neighbor items to the recommendation and user’s historical favorites nearest in latent space; not perfect causality, but intuitive evidence.
UI hook: add a “Why this?” expander to render these explanations.


## License

MIT (see `LICENSE`).
