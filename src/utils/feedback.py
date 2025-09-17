# src/utils/feedback.py
from .tune_als import append_feedback_csv


def record_feedback(user_id: int, movie_id: int, rating: float):
    append_feedback_csv(user_id, movie_id, rating)

