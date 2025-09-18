# Recommender Systems with PySpark

This project implements a **Movie Recommender System** using **PySpark ALS**. It features collaborative filtering, a cold-start fallback, and a Streamlit UI—all containerised with Docker.

---

## Quickstart

```bash
# Build environment
docker compose build --no-cache

# Download MovieLens data
docker compose run --rm app bash -lc "python src/data/download_movielens.py --dest data/raw"

# Train model (sample dataset)
docker compose run --rm app bash -lc "python -m src.models.train_als --ratings_csv data/sample/ratings_sample.csv --movies_csv data/sample/movies_sample.csv --model_dir models/als"

# Train full model
make train-full

# Run Streamlit app
make up
# → open http://localhost:8501
```

---

## Features

- ALS collaborative filtering (PySpark)  
- Hyperparameter tuning (`tune_als.py`)  
- Cold-start fallback: item–item cosine similarity  
- Precomputed top-N similarities for speed  
- Streamlit UI with user feedback logging  
- CI/CD via GitHub Actions  
- Optional Airflow retraining DAG

---

## CI Pipeline

- Linting + tests run automatically on push  
- Build/test Docker images  
- (Optional) Deploy step

Run locally:  
```bash
make ci
```

---

## Airflow (Optional)

Start Airflow service with:  
```bash
docker compose up airflow
```

Nightly retrain DAG in `dags/retrain.py`.

---

## Roadmap

- [x] ALS training + tuning  
- [x] Streamlit UI  
- [x] Cold-start fallback  
- [x] CI/CD pipeline  
- [x] Airflow retrain DAG  
- [x] MLflow integration  
- [ ] Additional datasets

---

## Repo Structure

```
src/
  data/
    download_movielens.py
  models/
    train_als.py
    tune_als.py
  app.py
data/
  raw/
  sample/
models/
docker-compose.yml
Dockerfile
Makefile
README.md
```
