# Recommender systems with Pyspark

End-to-end movie recommendation MVP using PySpark (ALS) and Streamlit.

## Features
- Data ingestion for MovieLens (small) + optional IMDB.
- Feature building (user–item ratings, movie metadata).
- Model training with **ALS** (explicit ratings).
- Fast item–item recommendations using ALS item factors.
- Streamlit UI: input favourite titles and get recommendations.
- Dockerfile for reproducible environment.

## Quickstart

## 🚀 Run with Docker (recommended)

```bash
# build the image
docker compose build

# start the app (http://localhost:8501)
docker compose up
```

This uses **OpenJDK 17** inside the container and the PySpark wheel ships Spark, so no local Java/Spark setup is required. Your repo folder is mounted into `/app`, so edits on your Mac reload instantly.

### Train the model in Docker
```bash
# terminal 1: run an interactive shell in the container
docker compose run --rm app bash

# inside the container:
python src/data/download_movielens.py --dest data/raw
python src/models/train_als.py \
  --ratings_csv data/sample/ratings_sample.csv \
  --movies_csv  data/sample/movies_sample.csv \
  --model_dir   models/als
exit
```

### Useful Docker commands
```bash
docker compose ps
docker compose logs -f
docker compose down
```

> Tip: If port 8501 is busy, change it in `docker-compose.yml` to `HOSTPORT:8501`.

### 1) Prerequisites
- **Python 3.10** (recommended for PySpark 3.5.x compatibility)
- Java 8+ JRE/JDK (Spark requires a JVM)
- (Optional) Docker

### 2) Run via Docker (recommended)
```bash
docker build -t pyspark-recsys .
docker run -p 8501:8501 pyspark-recsys
```

### 3) (Alternative) Create & activate a virtual environment
```bash
cd Recommender_systems_with_Pyspark
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) (Option A) Use included tiny sample data
This repo ships with a tiny `data/sample/` subset for an offline demo.

### 3) (Option B) Download MovieLens (small)
```bash
python src/data/download_movielens.py --dest data/raw
```

### 4) Train the model
```bash
python src/models/train_als.py --ratings_csv data/sample/ratings_sample.csv --movies_csv data/sample/movies_sample.csv --model_dir models/als
```

### 5) Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### 6) Docker (optional)
```bash
docker build -t pyspark-recsys .
docker run -p 8501:8501 pyspark-recsys
```

## Repository layout
```
app/                    # Streamlit UI
config/                 # YAML config
data/raw/               # Raw datasets (gitignored)
data/processed/         # Cleaned/parquet (gitignored)
data/sample/            # Tiny CSV sample for offline demo
docs/                   # Notes, design docs, MVP outline
models/                 # Saved ALS model (gitignored)
notebooks/              # EDA sketches (gitignored *.ipynb)
src/                    # Python package-ish code
  data/                 # Ingestion scripts
  features/             # Feature building / joins
  models/               # Training & inference
  utils/                # Spark session, IO helpers
tests/                  # Minimal tests
```

## Data sources
- [MovieLens](https://grouplens.org/datasets/movielens/)
- [IMDb datasets](https://datasets.imdbws.com/)

> ⚠️ Check licenses/terms before redistribution.

## Make it yours
- Swap in your data source,
- Extend the app to login/save user profiles,
- Or wire a scheduler (Airflow) to refresh models.

## License
MIT