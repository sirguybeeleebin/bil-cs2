.PHONY: format lint test m

format:
	poetry run autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive ./
	poetry run isort ./
	poetry run ruff format ./
	
lint:
	poetry run ruff check ./
	
test:
	poetry run pytest auth -v --disable-warnings -p no:cacheprovider --log-cli-level=INFO
	
prune:
	docker compose stop
	docker container prune -f
	docker volume prune -f	
	docker volume rm bil-cs2_postgres_data

up:
	docker compose up --build

