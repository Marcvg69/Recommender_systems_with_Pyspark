import streamlit as st
import pandas as pd
from src.models.infer_recommendations import recommend_for_favorites
from src.utils.io import load_config

st.set_page_config(page_title="PySpark Recommender", layout="centered")

st.title("🎬 Movie Recommender (PySpark ALS)")

cfg = load_config()
movies_csv = cfg["paths"]["movies_csv"]
model_dir = cfg["paths"]["model_dir"]

st.sidebar.header("Data & Model")
st.sidebar.write(f"Movies CSV: `{movies_csv}`")
st.sidebar.write(f"Model dir: `{model_dir}`")

st.write("Type a few favourite movie titles and press **Recommend**.")

# Load movie titles for selection
@st.cache_data
def load_titles(path):
    df = pd.read_csv(path)
    return sorted(df["title"].dropna().unique().tolist())

titles = load_titles(movies_csv)

favorites = st.multiselect("Select 2–5 favourites:", options=titles, default=titles[:2] if titles else [])

top_k = st.slider("How many recommendations?", 5, 30, 10, step=1)

if st.button("Recommend"):
    if len(favorites) < 1:
        st.warning("Please add at least one favourite title.")
    else:
        with st.spinner("Computing recommendations..."):
            try:
                recs = recommend_for_favorites(model_dir=model_dir, movies_csv=movies_csv, favorites=favorites, top_k=top_k)
                st.subheader("Recommendations")
                df = pd.DataFrame(recs, columns=["title","similarity"])
                st.dataframe(df)
            except Exception as e:
                st.error(str(e))