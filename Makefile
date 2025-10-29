.PHONY: format lint test m

SRC := ./

format:
	poetry run autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive $(SRC)
	poetry run isort $(SRC)
	poetry run ruff format $(SRC)
	
lint:
	poetry run ruff check $(SRC)
	
test:
	DJANGO_SETTINGS_MODULE=config.settings poetry run pytest -v --disable-warnings -p no:cacheprovider --log-cli-level=INFO
	
prune:
	docker compose stop
	docker container prune -f
	docker volume prune -f	
	docker volume rm bil-cs2_postgres_data
	docker volume rm bil-cs2_mongo_data

up:
	docker compose up --build