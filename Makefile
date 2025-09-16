.PHONY: build train train-full up down logs bash clean

build:
	docker compose build --no-cache

train:
	# Train on sample (fast)
	docker compose run --rm app bash -lc "\
	 python -m src.models.train_als \
	   --ratings_csv data/sample/ratings_sample.csv \
	   --movies_csv  data/sample/movies_sample.csv \
	   --model_dir   models/als"

train-full:
	# Train on full MovieLens small
	docker compose run --rm app bash -lc "\
	 python -m src.models.train_als \
	   --ratings_csv data/raw/ml-latest-small/ratings.csv \
	   --movies_csv  data/raw/ml-latest-small/movies.csv \
	   --model_dir   models/als"

up:
	docker compose up

down:
	docker compose down -v --remove-orphans

logs:
	docker compose logs -f app

bash:
	docker compose run --rm app bash

clean:
	rm -rf models/als models/als_best
