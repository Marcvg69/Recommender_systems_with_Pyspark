# MVP (End of Day 1) – Outline

**Project:** Recommender systems with Pyspark  
**Owner:** Individual challenge  
**Date:** 2025-09-15

## 1. Problem statement (user perspective)
Help a user discover new movies by inputting a few favourites and receiving 10 similar recommendations instantly via a simple web UI.

## 2. Scope for MVP
- Backend model: PySpark ALS (explicit ratings) trained on MovieLens small (or tiny sample for demo).
- Inference: item–item similarity from ALS item factors to recommend for arbitrary favourites (no need for an existing user id).
- Frontend: Streamlit app with multiselect for titles and a Recommend button.
- Packaging: requirements.txt + virtualenv instructions; optional Dockerfile.
- Out of scope (for now): user auth, persistence, incremental refresh, Airflow orchestration, cloud deploy.

## 3. Success criteria
- Repo installs in <10 minutes on a clean machine (Python 3.10, Java available).
- `train_als.py` runs and saves a model locally.
- Streamlit app returns recommendations for at least 2 input favourites without errors.
- Clean README and repo structure.

## 4. Data plan
- Start with MovieLens small (download script).
- Provide tiny `data/sample/` csvs to make the demo work offline.
- Optional: later enrich with IMDb metadata (genres, year).

## 5. Technical architecture (MVP)
- **Storage:** CSV in `data/`, saved Spark model in `models/als`.
- **Compute:** Local Spark session via PySpark.
- **Model:** ALS (rank=50, reg=0.05, iter=10).
- **Serving:** Streamlit triggers Python inference that loads ALS model and computes cosine similarity between item factors and favourite items.

## 6. Risks & mitigations
- **Java/Spark setup issues:** Provide Dockerfile as fallback.
- **Title mismatches:** Autocomplete from movies list.
- **Cold start for niche titles:** Fall back to genre-based popularity (v2).

## 7. Next steps (Days 2–5)
- Day 2: Add genre/year awareness (hybrid scoring = ALS similarity + content similarity).
- Day 3: Persist user sessions; evaluation notebook; plots.
- Day 4: Containerise + optional Render deploy.
- Day 5: Polish UI, write 10-min presentation, tests and linting.

---

## Timeline (today)
- ✅ Repo skeleton + README + sample data + training & inference scripts.
- ✅ Streamlit MVP that accepts favourites and returns top-N.
- ✅ Submit this outline to coach for feedback.