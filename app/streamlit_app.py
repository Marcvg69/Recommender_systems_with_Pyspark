# app/streamlit_app.py
# UI for: Known user (ALS) + New user (cold-start)
# - Fixes title mapping and warns on dataset mismatch
# - Restores "Add review" (logs to data/feedback/reviews.csv)

import os
import pathlib
from datetime import datetime

import pandas as pd
import streamlit as st

# Lazy Spark imports (so the page renders even if Spark isn't installed)
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.ml.recommendation import ALSModel
    SPARK_OK = True
except Exception:
    SPARK_OK = False

# Similarity helpers (package path first, then local fallback)
try:
    from src.models.similarity import (
        build_item_user_matrix,
        recommend_for_new_user,
        popular_movies_baseline,
        weighted_popularity,
    )
except Exception:
    from similarity import (  # type: ignore
        build_item_user_matrix,
        recommend_for_new_user,
        popular_movies_baseline,
        weighted_popularity,
    )

# -------- Config / Defaults --------
DEFAULT_RATINGS_CSV = os.environ.get("RATINGS_CSV", "data/sample/ratings_sample.csv")
DEFAULT_MOVIES_CSV  = os.environ.get("MOVIES_CSV",  "data/sample/movies_sample.csv")
DEFAULT_MODEL_DIR   = os.environ.get("ALS_MODEL_DIR", "models/als_best")
TOP_N_DEFAULT       = 10

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender (PySpark ALS + Cold-Start Fallback)")

# -------- Sidebar controls --------
ratings_csv = st.sidebar.text_input("Ratings CSV", value=DEFAULT_RATINGS_CSV)
movies_csv  = st.sidebar.text_input("Movies CSV",  value=DEFAULT_MOVIES_CSV)
als_model_dir = st.sidebar.text_input("ALS model directory", value=DEFAULT_MODEL_DIR)
top_n = int(st.sidebar.number_input("Top-N", min_value=1, max_value=50, value=TOP_N_DEFAULT, step=1))

# -------- Data loading (cached) --------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    ratings_df = load_csv(ratings_csv)
    movies_df  = load_csv(movies_csv)
except Exception as e:
    st.error(f"Could not read CSV files: {e}")
    st.stop()

# Normalize types early
if "userId" in ratings_df.columns:
    ratings_df["userId"] = pd.to_numeric(ratings_df["userId"], errors="coerce").astype("Int64")
if "movieId" in ratings_df.columns:
    ratings_df["movieId"] = pd.to_numeric(ratings_df["movieId"], errors="coerce").astype("Int64")
if "movieId" in movies_df.columns:
    movies_df["movieId"] = pd.to_numeric(movies_df["movieId"], errors="coerce").astype("Int64")

# Title maps
title2id = dict(zip(movies_df["title"].astype(str), movies_df["movieId"].astype(int)))
id2title = dict(zip(movies_df["movieId"].astype(int), movies_df["title"].astype(str)))

# -------- Spark helpers --------
@st.cache_resource(show_spinner=False)
def get_spark():
    if not SPARK_OK:
        return None
    try:
        return (
            SparkSession.builder
            .appName("RecommenderApp")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_als_model(model_dir: str):
    return ALSModel.load(model_dir)

def recommend_for_known_user_als(spark, model, user_id: int, movies_csv_path: str, k: int) -> pd.DataFrame:
    """Spark ALS inference + robust title mapping."""
    users = spark.createDataFrame([(int(user_id),)], ["userId"])
    recs_df = (
        model.recommendForUserSubset(users, k)
             .select("userId", F.explode("recommendations").alias("rec"))
             .select("userId", F.col("rec.movieId").cast("int").alias("movieId"),
                               F.col("rec.rating").alias("score"))
    )
    movies_sdf = (
        spark.read.option("header", True).option("inferSchema", True).csv(movies_csv_path)
             .withColumn("movieId", F.col("movieId").cast("int"))
    )
    out = recs_df.join(movies_sdf, on="movieId", how="left")
    pdf = out.orderBy(F.desc("score")).toPandas()

    # Fallback title mapping (pandas) for any still-missing titles
    if "title" not in pdf.columns:
        pdf["title"] = pdf["movieId"].map(id2title)
    else:
        null_mask = pdf["title"].isna()
        if null_mask.any():
            pdf.loc[null_mask, "title"] = pdf.loc[null_mask, "movieId"].map(id2title)

    # Warn if most titles are missing → likely dataset mismatch
    miss_ratio = float(pdf["title"].isna().mean()) if "title" in pdf.columns else 1.0
    if miss_ratio > 0.5:
        st.warning(
            "More than half of titles are missing. "
            "Your ALS model and Movies CSV likely come from different datasets.\n"
            "Tip: use *both* from ml-latest-small, or *both* from ml-25m."
        )

    return pdf[["movieId", "title", "score"]]

# -------- Similarity artifacts --------
@st.cache_resource(show_spinner=True)
def get_similarity_artifacts(ratings_csv_path: str):
    ratings = load_csv(ratings_csv_path)
    return build_item_user_matrix(ratings)

# -------- Simple feedback sink --------
def _append_feedback(user_or_mode: str, movie_id: int, rating: float, comment: str):
    out_dir = pathlib.Path("data/feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "reviews.csv"
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = pd.DataFrame([{
        "ts_utc": now,
        "mode": user_or_mode,
        "movieId": int(movie_id),
        "rating": float(rating),
        "comment": comment.strip(),
    }])
    if out_csv.exists():
        row.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        row.to_csv(out_csv, index=False)

# -------- UI --------
mode = st.radio("Mode", ["Known user (ALS)", "New user (cold-start)"])

# Keep last recommendations in session (for feedback UI)
if "last_recs" not in st.session_state:
    st.session_state.last_recs = None
if "last_user" not in st.session_state:
    st.session_state.last_user = None

if mode == "Known user (ALS)":
    spark = get_spark()
    if spark is None:
        st.error("Spark is not available. Install PySpark or run this app in the Docker container.")
        st.stop()
    try:
        model = load_als_model(als_model_dir)
    except Exception as e:
        st.error(
            f"ALS model not found at '{als_model_dir}'. "
            "Train and save a model first (see instructions below)."
        )
        st.stop()

    known_user_ids = sorted(list(set(int(u) for u in ratings_df["userId"].dropna().unique())))
    user_id = st.selectbox("Select a known userId", known_user_ids, index=0 if known_user_ids else None)

    if st.button("Recommend (ALS)"):
        try:
            recs = recommend_for_known_user_als(spark, model, int(user_id), movies_csv, k=top_n)
            st.session_state.last_recs = recs
            st.session_state.last_user = int(user_id)
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)
        except Exception as e:
            st.error(f"ALS recommendation failed: {e}")

    # --- Add review UI (restored) ---
    if st.session_state.last_recs is not None:
        st.subheader("💬 Add review / feedback")
        recs = st.session_state.last_recs
        choices = list(zip(recs["title"].fillna("(unknown title)"), recs["movieId"]))
        sel_title = st.selectbox("Select a movie to review", [t for t, _ in choices])
        sel_id = dict(choices).get(sel_title)
        stars = st.slider("Your rating", min_value=0.5, max_value=5.0, step=0.5, value=4.0)
        comment = st.text_input("Optional comment")
        if st.button("Save feedback"):
            _append_feedback(f"user:{st.session_state.last_user}", sel_id, stars, comment)
            st.success("Saved feedback to data/feedback/reviews.csv")

else:
    st.write("Pick a few movies you like to seed cold-start recommendations:")
    sample_titles = movies_df["title"].astype(str).tolist()
    seed_titles = st.multiselect("Type to search titles", sample_titles[:5000])
    seed_ids = [title2id[t] for t in seed_titles if t in title2id]

    item_user_mat, movie2idx, idx2movie = get_similarity_artifacts(ratings_csv)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Recommend (cosine similarity)"):
            if not seed_ids:
                st.warning("Select at least one liked movie.")
            else:
                rec_ids = recommend_for_new_user(
                    liked_movie_ids=seed_ids,
                    item_user_mat=item_user_mat,
                    movie2idx=movie2idx,
                    idx2movie=idx2movie,
                    top_n=top_n,
                    exclude=seed_ids,
                )
                rec_df = movies_df[movies_df["movieId"].isin(rec_ids)][["movieId", "title"]].copy()
                order_map = {mid: i for i, mid in enumerate(rec_ids)}
                rec_df["__order"] = rec_df["movieId"].map(order_map)
                rec_df = rec_df.sort_values("__order").drop(columns="__order")
                st.session_state.last_recs = rec_df
                st.session_state.last_user = "cold-start"
                st.subheader("Recommendations (cosine)")
                st.dataframe(rec_df, use_container_width=True)

    with col2:
        if st.button("Popular baseline"):
            pop_ids = popular_movies_baseline(ratings_df, top_n=top_n)
            pop_df = movies_df[movies_df["movieId"].isin(pop_ids)][["movieId", "title"]].copy()
            order_map = {mid: i for i, mid in enumerate(pop_ids)}
            pop_df["__order"] = pop_df["movieId"].map(order_map)
            pop_df = pop_df.sort_values("__order").drop(columns="__order")
            st.session_state.last_recs = pop_df
            st.session_state.last_user = "popular"
            st.subheader("Popular movies (mean rating + support)")
            st.dataframe(pop_df, use_container_width=True)

    with col3:
        if st.button("Weighted popularity (IMDB-style)"):
            wr_ids = weighted_popularity(ratings_df, top_n=top_n)
            wr_df = movies_df[movies_df["movieId"].isin(wr_ids)][["movieId", "title"]].copy()
            order_map = {mid: i for i, mid in enumerate(wr_ids)}
            wr_df["__order"] = wr_df["movieId"].map(order_map)
            wr_df = wr_df.sort_values("__order").drop(columns="__order")
            st.session_state.last_recs = wr_df
            st.session_state.last_user = "weighted_pop"
            st.subheader("Popular movies (weighted)")
            st.dataframe(wr_df, use_container_width=True)

    # Feedback for cold-start outputs
    if st.session_state.last_recs is not None:
        st.subheader("💬 Add review / feedback")
        recs = st.session_state.last_recs
        choices = list(zip(recs["title"].fillna("(unknown title)"), recs["movieId"]))
        sel_title = st.selectbox("Select a movie to review", [t for t, _ in choices], key="sel_cs_title")
        sel_id = dict(choices).get(sel_title)
        stars = st.slider("Your rating", min_value=0.5, max_value=5.0, step=0.5, value=4.0, key="cs_stars")
        comment = st.text_input("Optional comment", key="cs_comment")
        if st.button("Save feedback", key="cs_save"):
            _append_feedback(f"mode:{st.session_state.last_user}", sel_id, stars, comment)
            st.success("Saved feedback to data/feedback/reviews.csv")

st.caption("Tip: use matching datasets for Ratings/Movies and the ALS model (e.g., both from ml-25m).")
