import os
import pandas as pd
import streamlit as st

from src.models.infer_recommendations import recommend_for_favorites

# Defaults: full dataset + current ALS model
MOVIES_CSV_DEFAULT = "data/raw/ml-latest-small/movies.csv"
MODEL_DIR_DEFAULT = "models/als"

st.set_page_config(page_title="Movie Recommender (PySpark ALS)", layout="wide")

@st.cache_data
def load_titles(csv_path: str):
    """Read titles from a MovieLens-style CSV (must have a 'title' column)."""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    if "title" not in df.columns:
        return []
    return df["title"].dropna().astype(str).unique().tolist()

def main():
    st.title("🎬 Movie Recommender (PySpark ALS)")
    st.write("Type a few favourite titles and press **Recommend**.")

    with st.sidebar:
        st.header("Data & Model")
        movies_csv = st.text_input("Movies CSV:", MOVIES_CSV_DEFAULT)
        model_dir  = st.text_input("Model dir:",  MODEL_DIR_DEFAULT)

        # quick sanity hints
        if not os.path.exists(movies_csv):
            st.error(f"Movies CSV not found: {movies_csv}")
        if not os.path.isdir(model_dir):
            st.warning(f"Model directory not found: {model_dir}. Did you run training?")

    titles = load_titles(movies_csv)

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
        if not os.path.isdir(model_dir):
            st.error("Model directory missing. Train first (see Makefile: `make train-full`).")
            return

        # ✅ Correct order: (model_dir, movies_csv, favorites, k)
        recs = recommend_for_favorites(model_dir, movies_csv, selected, k)

        # Normalise to DataFrame for display
        if isinstance(recs, pd.DataFrame):
            df = recs
        elif isinstance(recs, list):
            df = pd.DataFrame(recs)
        else:
            df = pd.DataFrame(recs)

        # Do not recommend what user already selected (if column exists)
        if "title" in df.columns:
            df = df[~df["title"].isin(selected)]

        st.subheader("Recommendations")
        st.dataframe(df.reset_index(drop=True))

if __name__ == "__main__":
    main()
