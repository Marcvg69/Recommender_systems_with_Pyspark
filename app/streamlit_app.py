# streamlit_app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.recommendation import ALSModel

# --- Paths (adjust if your repo differs)
RATINGS_CSV = "data/raw/ratings.csv"                 # base ratings
MOVIES_CSV  = "data/raw/movies.csv"                  # movie metadata (movieId,title,genres)
MODEL_DIR   = "models/als_best"                      # trained model output
SIMS_DIR    = "data/processed/item_sims"             # precomputed item sims (fallback)
FEEDBACK_CSV= "data/feedback/ratings_feedback.csv"   # appended feedback

# --- Ensure folders exist
Path("data/feedback").mkdir(parents=True, exist_ok=True)

# --- Spark session (reuse)
@st.cache_resource
def get_spark():
    builder = SparkSession.builder.appName("recsys_app")
    # If you need to point Spark to a specific Java, ensure JAVA_HOME is set in your shell
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

spark = get_spark()

# --- Helpers
def read_csv_df(spark, path, schema):
    return spark.read.option("header", True).schema(schema).csv(path)

def load_model():
    return ALSModel.load(MODEL_DIR)

def load_dataframes():
    ratings_schema = T.StructType([
        T.StructField("userId", T.IntegerType(), False),
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("rating", T.FloatType(), False),
        T.StructField("timestamp", T.LongType(), True),
    ])
    movies_schema = T.StructType([
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("title", T.StringType(), True),
        T.StructField("genres", T.StringType(), True),
    ])
    ratings = read_csv_df(spark, RATINGS_CSV, ratings_schema).cache()
    movies  = read_csv_df(spark, MOVIES_CSV,  movies_schema).cache()
    return ratings, movies

def append_feedback_csv(user_id: int, movie_id: int, rating: float, path: str = FEEDBACK_CSV):
    p = Path(path)
    header_needed = not p.exists()
    with p.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("userId,movieId,rating,timestamp\n")
        import time
        f.write(f"{user_id},{movie_id},{rating},{int(time.time())}\n")

def recommend_with_fallback(spark, model_dir, sims_dir, for_user: int, ratings_df, k: int = 10):
    """
    Try ALS recommendations; if the user has no factors, fallback to item-item neighbors aggregated from
    the user's history. If no history at all, return globally similar-to-popular.
    """
    model = ALSModel.load(model_dir)
    user_df = spark.createDataFrame([(for_user,)], "userId INT")

    # Try ALS user subset
    try:
        recs = model.recommendForUserSubset(user_df, k)
        out = recs.select("userId", F.explode("recommendations").alias("rec")) \
                  .select("userId", F.col("rec.movieId").alias("movieId"),
                          F.col("rec.rating").alias("score"))
        return out
    except Exception:
        pass

    # Fallback using item-item similarities
    sims = spark.read.parquet(sims_dir)  # movieId, neighborId, similarity, rank
    history = ratings_df.filter(F.col("userId") == F.lit(for_user)) \
                        .select("movieId", "rating")
    if history.count() == 0:
        # no history: most-similar-to-popular
        agg = sims.groupBy("neighborId").agg(F.avg("similarity").alias("score"))
        return agg.orderBy(F.desc("score")).limit(k) \
                  .withColumn("userId", F.lit(for_user)) \
                  .select("userId", F.col("neighborId").alias("movieId"), "score")

    joined = sims.join(history, on="movieId", how="inner") \
                 .withColumn("weighted", F.col("similarity") * F.col("rating"))
    scores = joined.groupBy("neighborId").agg(
        F.sum("weighted").alias("score"),
        F.max("similarity").alias("max_sim")
    ).orderBy(F.desc("score"))
    return scores.select(F.lit(for_user).alias("userId"),
                         F.col("neighborId").alias("movieId"),
                         "score").limit(k)

# --- UI
st.title("🎬 PySpark Recommender")

# Load data
ratings_df, movies_df = load_dataframes()

# Sidebar: user selection
user_ids = [r.userId for r in ratings_df.select("userId").distinct().orderBy("userId").limit(5000).collect()]
default_user = user_ids[0] if user_ids else 1
user_id = st.sidebar.number_input("User ID", min_value=1, value=default_user, step=1)

k = st.sidebar.slider("How many recommendations?", 5, 30, 10)

# Show recent ratings for context
st.subheader("Recent ratings for this user")
recent = ratings_df.filter(F.col("userId") == F.lit(int(user_id))) \
                   .orderBy(F.desc("timestamp")).limit(10) \
                   .join(movies_df, on="movieId", how="left") \
                   .select("movieId", "title", "rating")
st.dataframe(recent.toPandas() if recent.count() else pd.DataFrame(columns=["movieId","title","rating"]))

# Load model and recommend
if not Path(MODEL_DIR).exists():
    st.warning("No trained model found at models/als_best/. Please run the tuner first.")
else:
    try:
        recs = recommend_with_fallback(spark, MODEL_DIR, SIMS_DIR, int(user_id), ratings_df, k=k) \
               .join(movies_df, on="movieId", how="left") \
               .select("movieId", "title", "score")
        st.subheader("Recommendations")
        st.dataframe(recs.toPandas())
    except Exception as e:
        st.error(f"Could not produce recommendations: {e}")

# Feedback widget
st.subheader("Leave a quick rating")
with st.form("feedback"):
    movie_title = st.text_input("Movie title (exact match)", "")
    rating_val  = st.slider("Your rating", 0.5, 5.0, 3.0, 0.5)
    submitted = st.form_submit_button("Submit")
    if submitted:
        if not movie_title.strip():
            st.warning("Please enter a movie title.")
        else:
            # lookup movieId by title
            movie_row = movies_df.filter(F.col("title") == F.lit(movie_title)).select("movieId").limit(1).collect()
            if not movie_row:
                st.error("Title not found in movies list.")
            else:
                m_id = int(movie_row[0]["movieId"])
                append_feedback_csv(int(user_id), m_id, float(rating_val))
                st.success("Thanks! Saved your feedback to data/feedback/ratings_feedback.csv")
