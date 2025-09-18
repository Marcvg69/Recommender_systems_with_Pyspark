import argparse
from typing import List
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F, types as T
from src.utils.spark import get_spark

def cosine(u, v):
    # u and v are arrays
    import math
    if u is None or v is None:
        return None
    dot = sum(a*b for a,b in zip(u,v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot/(nu*nv)

def recommend_for_favorites(model_dir: str, movies_csv: str, favorites: List[str], top_k: int = 10):
    spark = get_spark("infer_recommendations")
    model = ALSModel.load(model_dir)

    movies = spark.read.csv(movies_csv, header=True, inferSchema=True)
    # Expect columns: movieId,title
    fav_df = movies.where(F.col("title").isin(favorites)).select("movieId","title").cache()
    fav_ids = [r.movieId for r in fav_df.collect()]
    if not fav_ids:
        raise ValueError("None of the provided favorite titles match movies in the dataset.")

    # itemFactors schema: (id, features)
    items = model.itemFactors

    # gather feature vectors for favorite items
    fav_vecs = items.where(F.col("id").isin(fav_ids)).select("id","features").collect()
    fav_map = {r["id"]: r["features"] for r in fav_vecs}

    # UDF for max cosine similarity to any favorite
    cos_udf = F.udf(lambda feat: max(
        (cosine(feat, fv) for fv in fav_map.values()), default=0.0
    ), T.FloatType())

    scored = items.withColumn("score", cos_udf(F.col("features")))

    # Join back to movie titles
    recs = scored.join(movies, scored.id == movies.movieId, "inner") \
                 .select(movies.movieId, "title", "score") \
                 .orderBy(F.desc("score"))

    # exclude favorites
    recs = recs.where(~F.col("movieId").isin(fav_ids)).limit(top_k)

    out = [ (r["title"], float(r["score"])) for r in recs.collect() ]
    spark.stop()
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/als")
    parser.add_argument("--movies_csv", required=True)
    parser.add_argument("--favorites", nargs="+", required=True, help="List of movie titles")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    recs = recommend_for_favorites(args.model_dir, args.movies_csv, args.favorites, args.top_k)
    for title, score in recs:
        print(f"{title}\t{score:.4f}")