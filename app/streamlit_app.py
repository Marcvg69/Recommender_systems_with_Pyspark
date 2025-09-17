# app/streamlit_app.py
from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd
import pyspark.sql.functions as F
import streamlit as st
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, Row, SparkSession


# -----------------------------
# Spark helpers
# -----------------------------
def get_spark(app_name: str = "recsys", memory: str = "2g") -> SparkSession:
    """
    Create or get a local SparkSession suitable for the container and CI.

    Notes
    -----
    * We keep it simple: local[*], no Hive, and a small driver memory cap.
    * UI is disabled to reduce noise in container logs.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )
    return spark


# -----------------------------
# Data loading
# -----------------------------
def read_movies(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, inferSchema=True)


def read_ratings(spark: SparkSession, base_path: str) -> DataFrame:
    raw = spark.read.csv(
        os.path.join(base_path, "ratings.csv"), header=True, inferSchema=True
    )
    fb_path = os.path.join("data", "feedback", "ratings_feedback.csv")
    if Path(fb_path).exists():
        fb = spark.read.csv(fb_path, header=True, inferSchema=True)
        raw = raw.unionByName(fb, allowMissingColumns=True)
    return raw


def load_model(model_dir: str) -> ALSModel | None:
    try:
        return ALSModel.load(model_dir)
    except Exception:
        return None


# -----------------------------
# Recommenders
# -----------------------------
def recommend_with_fallback(
    spark: SparkSession,
    model_dir: str,
    movies_df: DataFrame,
    ratings_df: DataFrame,
    for_user: int,
    k: int = 10,
) -> DataFrame:
    """
    Try ALS recommendations. If the user has no factors, fall back to a very
    simple popularity-based list computed from ratings.

    Returns a Spark DataFrame with columns: movieId, title, score.
    """
    model = load_model(model_dir)

    if model is not None:
        # Is the user present in the trained user factors?
        has_factors = (
            model.userFactors.filter(F.col("id") == F.lit(for_user)).limit(1).count() > 0
        )
        if has_factors:
            users = spark.createDataFrame([Row(userId=for_user)])
            recs = model.recommendForUserSubset(users, k)
            exploded = recs.select(
                F.explode("recommendations").alias("rec")
            ).select(
                F.col("rec.movieId").alias("movieId"),
                F.col("rec.rating").alias("score"),
            )
            out = (
                exploded.join(movies_df, on="movieId", how="left")
                .select("movieId", "title", "score")
                .orderBy(F.desc("score"))
            )
            return out

    # Fallback: most popular (by count and avg rating)
    agg = (
        ratings_df.groupBy("movieId")
        .agg(
            F.count(F.lit(1)).alias("cnt"),
            F.avg("rating").alias("avg"),
        )
        .withColumn("score", F.col("cnt") * F.col("avg"))
    )
    out = (
        agg.join(movies_df, on="movieId", how="left")
        .select("movieId", "title", "score")
        .orderBy(F.desc("score"))
        .limit(k)
    )
    return out


# -----------------------------
# Feedback writing
# -----------------------------
def append_feedback(user_id: int, movie_id: int, rating: float) -> Path:
    """
    Append a single feedback row to data/feedback/ratings_feedback.csv.
    Creates the directory and file header if missing.
    """
    out_dir = Path("data/feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ratings_feedback.csv"

    is_new = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["userId", "movieId", "rating", "timestamp"])
        writer.writerow([user_id, movie_id, rating, int(pd.Timestamp.now().timestamp())])
    return out_csv


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Movie Recs (ALS + fallback)", layout="wide")

st.title("🎬 Movie Recommendations")

spark = get_spark()

# Paths inside the repo/container
MOVIES_CSV = "data/raw/movies.csv"
RATINGS_BASE = "data/raw"
MODEL_DIR = "models/als_best"
SIMS_DIR = "data/processed/item_sims"  # kept for future use

# Load base data
movies_df = read_movies(spark, MOVIES_CSV)
ratings_df = read_ratings(spark, RATINGS_BASE)

# Sidebar controls
st.sidebar.header("Controls")
user_ids = [
    r.userId
    for r in (
        ratings_df.select("userId")
        .distinct()
        .orderBy("userId")
        .limit(5000)
        .collect()
    )
]
default_user = user_ids[0] if user_ids else 1
user_id = st.sidebar.number_input("User ID", min_value=1, value=default_user, step=1)
top_k = int(st.sidebar.slider("How many recommendations?", 5, 50, value=10, step=1))

# Show recent ratings for context
with st.expander("Recent ratings for the selected user"):
    recent = (
        ratings_df.filter(F.col("userId") == F.lit(int(user_id)))
        .orderBy(F.desc("timestamp"))
        .limit(25)
        .join(movies_df, on="movieId", how="left")
        .select("movieId", "title", "rating")
    )
    recent_pd = (
        recent.toPandas()
        if recent.count()
        else pd.DataFrame(columns=["movieId", "title", "rating"])
    )
    st.dataframe(recent_pd, use_container_width=True)

# Recommend
st.subheader("Top recommendations")
recs = recommend_with_fallback(
    spark=spark,
    model_dir=MODEL_DIR,
    movies_df=movies_df,
    ratings_df=ratings_df,
    for_user=int(user_id),
    k=top_k,
)
recs_pd = recs.toPandas()
st.dataframe(recs_pd, use_container_width=True)

# Quick feedback form
st.subheader("Leave a quick rating")
with st.form("quick_feedback"):
    movie_title = st.text_input("Movie title (exact match)", value="")
    score = st.slider("Your rating", min_value=0.5, max_value=5.0, step=0.5, value=4.5)
    submitted = st.form_submit_button("Submit")

if submitted:
    if not movie_title.strip():
        st.error("Please enter a movie title.")
    else:
        movie_row = (
            movies_df.filter(F.col("title") == F.lit(movie_title))
            .select("movieId")
            .limit(1)
            .collect()
        )
        if not movie_row:
            st.error("Title not found in movies list.")
        else:
            mid = int(movie_row[0]["movieId"])
            path = append_feedback(int(user_id), mid, float(score))
            st.success(
                f"Thanks! Saved your feedback to {path.as_posix()}"
            )

st.caption(
    "Tip: new feedback is appended immediately. The Airflow DAG can merge it and "
    "retrain ALS nightly."
)
