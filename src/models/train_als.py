import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

from src.utils.spark import get_spark

def train(ratings_csv: str, model_dir: str, rank: int = 50, reg: float = 0.1, max_iter: int = 10):
    spark: SparkSession = get_spark("train_als")

    # ratings.csv: userId,movieId,rating,timestamp
    df = spark.read.csv(ratings_csv, header=True, inferSchema=True) \
        .select(col("userId").cast("int"),
                col("movieId").cast("int"),
                col("rating").cast("float"))

    # simple split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        regParam=reg,
        maxIter=max_iter,
        nonnegative=True,
        coldStartStrategy="drop",
        implicitPrefs=False,
        seed=42,
    )

    model = als.fit(train_df)

    preds = model.transform(test_df)
    rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction").evaluate(preds)
    print(f"Test RMSE: {rmse:.4f}")

    # overwrite so re-training is easy during dev
    model.write().overwrite().save(model_dir)
    print(f"Model saved to {model_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratings_csv", required=True)
    p.add_argument("--movies_csv", required=False, help="Not used here but kept for CLI symmetry", default="")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--rank", type=int, default=50)
    p.add_argument("--reg", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=10)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args.ratings_csv, args.model_dir, rank=args.rank, reg=args.reg, max_iter=args.max_iter)
