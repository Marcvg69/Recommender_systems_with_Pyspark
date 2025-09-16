import streamlit as st
import pandas as pd

from src.models.infer_recommendations import recommend_for_favorites
from src.utils.io import load_movies_csv

# Defaults: use full dataset + your saved ALS model
MOVIES_CSV_DEFAULT = "data/raw/ml-latest-small/movies.csv"
MODEL_DIR_DEFAULT = "models/als"

st.set_page_config(page_title="Movie Recommender (PySpark ALS)", layout="wide")

@st.cache_data
def cached_titles(movies_csv: str):
    """Cache the titles list for a given CSV path."""
    df = load_movies_csv(movies_csv)
    if isinstance(df, pd.DataFrame) and "title" in df.columns:
        return df["title"].dropna().astype(str).unique().tolist()
    # Fallback if util returns a list already
    return sorted({str(x) for x in df})

def main():
    st.title("🎬 Movie Recommender (PySpark ALS)")
    st.write("Type a few favourite titles and press **Recommend**.")

    with st.sidebar:
        st.header("Data & Model")
        movies_csv = st.text_input("Movies CSV:", MOVIES_CSV_DEFAULT)
        model_dir = st.text_input("Model dir:", MODEL_DIR_DEFAULT)

    titles = cached_titles(movies_csv)

    selected = st.multiselect(
        "Select 2–5 favourites:",
        options=titles,
        max_selections=5,
    )
    k = st.slider("How many recommendations?", min_value=1, max_value=50, value=10)

    if st.button("Recommend", type="primary"):
        if len(selected) < 2:
            st.warning("Pick at least two favourites.")
            return

        # Use your existing inference helper (keeps internal logic intact)
        recs = recommend_for_favorites(movies_csv, model_dir, selected, k)

        # Be tolerant to return type (df / list[dict] / list[tuple])
        if hasattr(recs, "to_dict"):
            st.dataframe(recs)
        elif isinstance(recs, list):
            st.dataframe(pd.DataFrame(recs))
        else:
            st.write(recs)

if __name__ == "__main__":
    main()
