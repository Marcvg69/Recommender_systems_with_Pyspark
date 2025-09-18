"""
similarity.py
--------------
Cosine-similarity fallback for brand-new (cold-start) users + popularity baseline.
Pure pandas/numpy/scipy/sklearn — no Spark dependency.

Public API:
- build_item_user_matrix(ratings_df, user_col="userId", item_col="movieId", rating_col="rating")
- recommend_for_new_user(liked_movie_ids, item_user_mat=None, movie2idx=None, idx2movie=None,
                         ratings_df=None, top_n=10, exclude=None)
- popular_movies_baseline(ratings_df, movie_col="movieId", rating_col="rating",
                          min_count=50, top_n=10)
- weighted_popularity(ratings_df, movie_col="movieId", rating_col="rating",
                      m=50, C=None, top_n=10)
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

MovieId = Union[int, np.integer]
ItemUserMat = csr_matrix


def build_item_user_matrix(
    ratings_df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> Tuple[ItemUserMat, Dict[MovieId, int], List[MovieId]]:
    """
    Build sparse user x item matrix and lookup maps from a ratings DataFrame.
    Returns: (csr_matrix, movie2idx, idx2movie)
    """
    df = ratings_df[[user_col, item_col, rating_col]].dropna().copy()
    df[user_col] = df[user_col].astype("category")
    df[item_col] = df[item_col].astype("category")

    user_codes = df[user_col].cat.codes.values
    item_codes = df[item_col].cat.codes.values
    data = df[rating_col].astype(float).values

    n_users = df[user_col].cat.categories.size
    n_items = df[item_col].cat.categories.size
    mat = csr_matrix((data, (user_codes, item_codes)), shape=(n_users, n_items))

    idx2movie = list(df[item_col].cat.categories)
    try:
        idx2movie = [int(x) for x in idx2movie]
    except Exception:
        pass
    movie2idx = {mid: i for i, mid in enumerate(idx2movie)}
    return mat, movie2idx, idx2movie


def _cosine_vector_for_item(mat_T: csr_matrix, idx: int) -> np.ndarray:
    """Cosine similarity between one item (row idx of mat_T) and all items."""
    return cosine_similarity(mat_T[idx], mat_T).ravel()


def _aggregate_seed_similarities(
    item_user_mat: ItemUserMat,
    movie2idx: Dict[MovieId, int],
    seed_ids: Sequence[MovieId],
    exclude: Optional[Iterable[MovieId]] = None,
) -> np.ndarray:
    """Sum cosine similarity vectors for all seed movies; mask excluded items."""
    mat_T = item_user_mat.T  # items x users
    agg = None
    for mid in seed_ids:
        idx = movie2idx.get(mid)
        if idx is None:
            continue
        sims = _cosine_vector_for_item(mat_T, idx)
        sims[idx] = -1.0  # remove self
        agg = sims if agg is None else (agg + sims)

    if agg is None:
        return np.full(mat_T.shape[0], -np.inf, dtype=float)

    if exclude:
        for mid in exclude:
            ex_idx = movie2idx.get(mid)
            if ex_idx is not None and 0 <= ex_idx < agg.size:
                agg[ex_idx] = -np.inf
    return agg


def recommend_for_new_user(
    liked_movie_ids: Sequence[MovieId],
    item_user_mat: Optional[ItemUserMat] = None,
    movie2idx: Optional[Dict[MovieId, int]] = None,
    idx2movie: Optional[List[MovieId]] = None,
    ratings_df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    exclude: Optional[Iterable[MovieId]] = None,
) -> List[MovieId]:
    """
    Recommend movies for a brand-new user using item–item cosine similarity.
    Either pass prebuilt (item_user_mat, movie2idx, idx2movie) OR provide ratings_df.
    """
    if not liked_movie_ids:
        return []

    if item_user_mat is None or movie2idx is None or idx2movie is None:
        if ratings_df is None:
            raise ValueError("Provide matrix+lookups OR ratings_df to build them.")
        item_user_mat, movie2idx, idx2movie = build_item_user_matrix(ratings_df)

    # Always exclude the seeds themselves (plus any extra)
    excl = set(liked_movie_ids)
    if exclude:
        excl |= set(exclude)

    agg = _aggregate_seed_similarities(item_user_mat, movie2idx, liked_movie_ids, exclude=excl)
    order = np.argsort(agg)[::-1]

    recs: List[MovieId] = []
    for i in order:
        if not np.isfinite(agg[i]):
            break
        mid = idx2movie[i]
        recs.append(int(mid) if isinstance(mid, (np.integer,)) else mid)
        if len(recs) >= top_n:
            break
    return recs


def popular_movies_baseline(
    ratings_df: pd.DataFrame,
    movie_col: str = "movieId",
    rating_col: str = "rating",
    min_count: int = 50,
    top_n: int = 10,
) -> List[MovieId]:
    """Return top-N movie IDs by mean rating with minimum support."""
    g = (
        ratings_df[[movie_col, rating_col]]
        .dropna()
        .groupby(movie_col)[rating_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    g = g[g["count"] >= max(1, int(min_count))].sort_values(["mean", "count"], ascending=False)
    mids = g[movie_col].tolist()
    try:
        mids = [int(x) for x in mids]
    except Exception:
        pass
    return mids[: top_n]


def weighted_popularity(
    ratings_df: pd.DataFrame,
    movie_col: str = "movieId",
    rating_col: str = "rating",
    m: int = 50,
    C: Optional[float] = None,
    top_n: int = 10,
) -> List[MovieId]:
    """
    IMDB-style weighted rating:
      WR = (v/(v+m))*R + (m/(v+m))*C
      v = #ratings, R = avg rating, m = min votes, C = global mean rating
    """
    g = (
        ratings_df[[movie_col, rating_col]]
        .dropna()
        .groupby(movie_col)[rating_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    if C is None:
        C = ratings_df[rating_col].mean()
    g["wr"] = (g["count"] / (g["count"] + m)) * g["mean"] + (m / (g["count"] + m)) * C
    g = g[g["count"] >= 1].sort_values(["wr", "count"], ascending=False)
    mids = g[movie_col].tolist()
    try:
        mids = [int(x) for x in mids]
    except Exception:
        pass
    return mids[: top_n]
