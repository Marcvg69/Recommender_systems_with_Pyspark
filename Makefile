.PHONY: docker app

docker:
	docker build -t pyspark-recsys .
	docker run -p 8501:8501 pyspark-recsys

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down