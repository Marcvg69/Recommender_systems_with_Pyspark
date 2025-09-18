# src/models/tune_als.py
# ALS tuning + safe item–item similarity precompute.
# - Grid over rank/regParam, save best model to models/als_best/
# - Precompute top-N neighbours:
#     * driver (NumPy, chunked) for small corpora  -> avoids Spark OOM
#     * LSH (finite radius) for larger corpora
# - Cold-start fallback helper
# - Docker/laptop-friendly Spark settings

import argparse
import json
import os
from math import sqrt
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT

DEFAULT_MODEL_DIR = "models/als_best"
SIMS_OUT_DIR = "data/processed/item_sims"
FEEDBACK_CSV = "data/feedback/ratings_feedback.csv"


# ---------------------------
# IO helpers
# ---------------------------
def read_ratings(spark: SparkSession, ratings_csv: str) -> DataFrame:
    schema = T.StructType([
        T.StructField("userId", T.IntegerType(), False),
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("rating", T.FloatType(), False),
        T.StructField("timestamp", T.LongType(), True),
    ])
    return spark.read.option("header", True).schema(schema).csv(ratings_csv)


def train_val_split(ratings: DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[DataFrame, DataFrame]:
    return ratings.randomSplit([1 - val_ratio, val_ratio], seed=seed)


# ---------------------------
# ALS training & evaluation
# ---------------------------
def fit_and_score(train: DataFrame, val: DataFrame, rank: int, reg: float, max_iter: int,
                  alpha: float, implicit_prefs: bool) -> Tuple[ALSModel, float]:
    als = ALS(
        maxIter=max_iter,
        regParam=reg,
        rank=rank,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        implicitPrefs=implicit_prefs,
        alpha=alpha,
        nonnegative=True,
        seed=42,
    )
    model = als.fit(train)
    preds = model.transform(val)
    evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(preds)
    return model, rmse


def save_best(model: ALSModel, out_dir: str):
    p = Path(out_dir)
    if p.exists():
        import shutil; shutil.rmtree(p)
    model.save(out_dir)


# ---------------------------
# Item–item similarities
# ---------------------------
def _dense_vec_from_list(colname: str):
    return F.udf(lambda xs: Vectors.dense(xs), VectorUDT())


def precompute_item_sims_driver_numpy(
    spark: SparkSession,
    model: ALSModel,
    topN: int,
    sims_out_dir: str,
    batch_size: int = 256,
):
    """
    DRAM-friendly top-N cosine via NumPy on the driver, processed in batches.
    Works great for <= ~20k items (MovieLens-small is ~10k).
    """
    # Collect item factors to the driver (ordered by id for reproducibility)
    rows = (model.itemFactors
            .select(F.col("id").alias("movieId"), "features")
            .orderBy("movieId")
            .collect())
    ids = np.array([r["movieId"] for r in rows], dtype=np.int32)
    feats = np.array([r["features"] for r in rows], dtype=np.float32)  # (n_items, dim)

    # L2-normalise rows -> cosine = dot product
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Fm = feats / norms  # (n, d)

    n, d = Fm.shape
    results = []

    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        B = Fm[i0:i1]                          # (b, d)
        sims = B @ Fm.T                        # (b, n)
        # exclude self matches in this block
        for j, row_idx in enumerate(range(i0, i1)):
            sims[j, row_idx] = -np.inf

        # topN per row (partial sort)
        top_idx = np.argpartition(sims, -topN, axis=1)[:, -topN:]
        # order by similarity desc inside the topN slice
        ordered = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(ordered, axis=1)[:, ::-1]
        top_sorted_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sorted_sim = np.take_along_axis(ordered, order, axis=1)

        # collect rows
        for j in range(i1 - i0):
            src = int(ids[i0 + j])
            for rnk in range(topN):
                dst = int(ids[top_sorted_idx[j, rnk]])
                sim = float(top_sorted_sim[j, rnk])
                results.append((src, dst, sim, rnk + 1))

    # Write via Spark
    out_schema = T.StructType([
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("neighborId", T.IntegerType(), False),
        T.StructField("similarity", T.FloatType(), False),
        T.StructField("rank", T.IntegerType(), False),
    ])
    df = spark.createDataFrame(results, schema=out_schema)
    df.repartition(1).write.mode("overwrite").parquet(sims_out_dir)


def precompute_item_sims_lsh(
    spark: SparkSession,
    model: ALSModel,
    topN: int,
    sims_out_dir: str,
    bucket_length: float = 1.2,
    num_hash_tables: int = 3,
    cos_threshold: float = 0.60,
):
    """
    LSH with a FINITE radius derived from a cosine cutoff.
    """
    factors = model.itemFactors.withColumnRenamed("id", "movieId") \
        .withColumn("features_vec", _dense_vec_from_list("features")("features"))

    # normalise to unit vectors
    def _norm_vec(v):
        arr = v.toArray()
        n = float((arr ** 2).sum()) ** 0.5
        if n == 0.0:
            return Vectors.dense(arr)
        return Vectors.dense((arr / n).tolist())

    norm_udf = F.udf(_norm_vec, VectorUDT())
    normed = factors.withColumn("norm_vec", norm_udf(F.col("features_vec"))).persist()

    dist_threshold = float(sqrt(2.0 * (1.0 - float(cos_threshold))))

    lsh = BucketedRandomProjectionLSH(
        inputCol="norm_vec", outputCol="hashes",
        bucketLength=bucket_length, numHashTables=num_hash_tables
    )
    model_lsh = lsh.fit(normed)

    approx = model_lsh.approxSimilarityJoin(normed, normed, dist_threshold, distCol="dist") \
        .select(
            F.col("datasetA.movieId").alias("movieId"),
            F.col("datasetB.movieId").alias("neighborId"),
            F.col("dist")
        ).where(F.col("movieId") != F.col("neighborId"))

    cos_sim = approx.withColumn("similarity", 1 - (F.col("dist") ** 2) / F.lit(2.0))

    from pyspark.sql.window import Window
    w = Window.partitionBy("movieId").orderBy(F.desc("similarity"))
    topk = cos_sim.withColumn("rank", F.row_number().over(w)).where(F.col("rank") <= topN)

    topk.repartition(1).write.mode("overwrite").parquet(sims_out_dir)
    normed.unpersist()


def precompute_item_sims(
    spark: SparkSession,
    model: ALSModel,
    topN: int,
    sims_out_dir: str,
    method: str = "auto",
    **kwargs
):
    """
    method: "auto" | "driver" | "lsh"
    - auto uses driver for <= 20k items, else LSH
    """
    n_items = model.itemFactors.count()
    use_driver = (method == "driver") or (method == "auto" and n_items <= 20000)
    if use_driver:
        precompute_item_sims_driver_numpy(
            spark, model, topN, sims_out_dir,
            batch_size=int(kwargs.get("batch_size", 256)),
        )
    else:
        precompute_item_sims_lsh(
            spark, model, topN, sims_out_dir,
            bucket_length=float(kwargs.get("bucket_length", 1.2)),
            num_hash_tables=int(kwargs.get("num_hash_tables", 3)),
            cos_threshold=float(kwargs.get("cos_threshold", 0.60)),
        )


# ---------------------------
# Fallback recommender
# ---------------------------
def recommend_with_fallback(
    spark: SparkSession,
    als_model_dir: str = DEFAULT_MODEL_DIR,
    sims_dir: str = SIMS_OUT_DIR,
    for_user: Optional[int] = None,
    from_user_history: Optional[DataFrame] = None,
    k: int = 10,
) -> DataFrame:
    model = ALSModel.load(als_model_dir)
    if for_user is None:
        raise ValueError("for_user must be provided")

    user_df = spark.createDataFrame([(int(for_user),)], "userId INT")

    # Try ALS
    try:
        recs = model.recommendForUserSubset(user_df, k)
        out = recs.select("userId", F.explode("recommendations").alias("rec")) \
                  .select("userId", F.col("rec.movieId").alias("movieId"),
                          F.col("rec.rating").alias("score"))
        if out.count() > 0:
            return out
    except Exception:
        pass

    # Item–item fallback
    if Path(sims_dir).exists():
        sims = spark.read.parquet(sims_dir)  # movieId, neighborId, similarity, rank
        history = None
        if from_user_history is not None:
            history = from_user_history.filter(F.col("userId") == F.lit(int(for_user))) \
                                       .select("movieId", "rating")
        if history is None or history.count() == 0:
            agg = sims.groupBy("neighborId").agg(F.avg("similarity").alias("score"))
            return agg.orderBy(F.desc("score")).limit(k) \
                     .withColumn("userId", F.lit(int(for_user))) \
                     .select("userId", F.col("neighborId").alias("movieId"), "score")

        joined = sims.join(history, on="movieId", how="inner") \
                     .withColumn("weighted", F.col("similarity") * F.col("rating"))
        scores = joined.groupBy("neighborId") \
                       .agg(F.sum("weighted").alias("score"),
                            F.max("similarity").alias("max_sim")) \
                       .orderBy(F.desc("score"))
        return scores.select(F.lit(int(for_user)).alias("userId"),
                             F.col("neighborId").alias("movieId"),
                             "score").limit(k)

    # Popularity (as last resort)
    if from_user_history is not None:
        pop = from_user_history.groupBy("movieId") \
                               .agg(F.avg("rating").alias("score"), F.count("*").alias("n")) \
                               .orderBy(F.desc("score"), F.desc("n")) \
                               .limit(k) \
                               .withColumn("userId", F.lit(int(for_user))) \
                               .select("userId", "movieId", "score")
        return pop

    return spark.createDataFrame([], "userId INT, movieId INT, score DOUBLE")


# ---------------------------
# Feedback capture
# ---------------------------
def append_feedback_csv(user_id: int, movie_id: int, rating: float, path: str = FEEDBACK_CSV):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not p.exists()
    with p.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("userId,movieId,rating,timestamp\n")
        import time; f.write(f"{user_id},{movie_id},{rating},{int(time.time())}\n")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Tune ALS and precompute item-item similarities.")
    parser.add_argument("--ratings_csv", required=True)
    parser.add_argument("--model_out", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--regParams", default="0.01,0.05,0.1")
    parser.add_argument("--maxIter", type=int, default=15)
    parser.add_argument("--implicitPrefs", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)

    # Sims options
    parser.add_argument("--sims_topN", type=int, default=30)
    parser.add_argument("--sims_out_dir", default=SIMS_OUT_DIR)
    parser.add_argument("--sims_method", choices=["auto", "driver", "lsh"], default="auto")
    parser.add_argument("--driver_batch", type=int, default=256)
    parser.add_argument("--sim_threshold", type=float, default=0.60)
    parser.add_argument("--lsh_bucket_length", type=float, default=1.2)
    parser.add_argument("--lsh_num_tables", type=int, default=3)
    parser.add_argument("--skip_sims", action="store_true")

    # Spark tuning
    parser.add_argument("--shuffle_partitions", type=int, default=64)
    parser.add_argument("--driver_mem", type=str, default=None)
    parser.add_argument("--executor_mem", type=str, default=None)

    args = parser.parse_args()

    # Memory defaults (Docker env vars win if set)
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY") or args.driver_mem or "2g"
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY") or args.executor_mem or driver_mem

    spark = (
        SparkSession.builder
        .appName("tune_als")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    ratings = read_ratings(spark, args.ratings_csv).cache()
    train, val = train_val_split(ratings)

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    regs = [float(x) for x in args.regParams.split(",") if x.strip()]

    results = []
    best_rmse, best_model, best_cfg = float("inf"), None, None

    for r in ranks:
        for reg in regs:
            model, rmse = fit_and_score(train, val, r, reg, args.maxIter, args.alpha, args.implicitPrefs)
            results.append({"rank": r, "regParam": reg, "rmse": rmse})
            if rmse < best_rmse:
                best_rmse, best_model, best_cfg = rmse, model, {"rank": r, "regParam": reg}

    print("Grid results:", json.dumps(results, indent=2))
    print("Best:", json.dumps({"rmse": best_rmse, **best_cfg}, indent=2))

    save_best(best_model, args.model_out)
    print(f"Saved best ALS model to {args.model_out}")

    if not args.skip_sims:
        precompute_item_sims(
            spark, best_model, args.sims_topN, args.sims_out_dir,
            method=args.sims_method,
            batch_size=args.driver_batch,
            bucket_length=args.lsh_bucket_length,
            num_hash_tables=args.lsh_num_tables,
            cos_threshold=args.sim_threshold,
        )
        print(f"Wrote item-item similarities to {args.sims_out_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
