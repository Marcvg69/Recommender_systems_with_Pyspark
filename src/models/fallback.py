# src/models/fallback.py
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALSModel

from .tune_als import recommend_with_fallback  # reuse

def recommend_for_user_with_history(spark: SparkSession, user_id: int, ratings_df, k=10):
    return recommend_with_fallback(
        spark, for_user=user_id, from_user_history=ratings_df, k=k
    )
