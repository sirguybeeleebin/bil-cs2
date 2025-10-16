.PHONY: format lint test clickhouse-up postgres-up redis-up prune start-app

SRC := ./

format:
	poetry run autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive $(SRC)
	poetry run isort $(SRC)
	poetry run ruff format $(SRC)
	
lint:
	poetry run ruff check $(SRC)
	
test:
	poetry run pytest -v --disable-warnings -p no:cacheprovider

clickhouse-up:	
	docker run -d \
		--name clickhouse \
		-p 9000:9000 \
		-p 8123:8123 \
		-e CLICKHOUSE_DB=cs2_db \
		-e CLICKHOUSE_USER=cs2_user \
		-e CLICKHOUSE_PASSWORD=cs2_password \
		-v clickhouse_data:/var/lib/clickhouse \
		clickhouse/clickhouse-server:24.8

rabbitmq-up:
	docker run -d \
		--name rabbitmq \
		-p 5672:5672 \
		-p 15672:15672 \
		-e RABBITMQ_DEFAULT_USER=cs2_user \
		-e RABBITMQ_DEFAULT_PASS=cs2_password \
		rabbitmq:3-management

postgres-up:
	docker run -d \
		--name postgres \
		-p 5432:5432 \
		-e POSTGRES_DB=cs2_db \
		-e POSTGRES_USER=cs2_user \
		-e POSTGRES_PASSWORD=cs2_password \
		-v postgres_data:/var/lib/postgresql/data \
		postgres:15-alpine

redis-up:
	docker run -d --name redis -p 6379:6379 redis:7

prune:	
	docker container prune -f
	docker volume prune -f
	docker volume rm clickhouse_data -f
	docker volume rm postgres_data -f

start-app: migrate
	@echo "🛠 Running migrations..."
	
	@echo "🚀 Starting Celery worker..."
	export DJANGO_SETTINGS_MODULE=config.settings; \
	poetry run celery -A config worker -l info & \
	@echo "🚀 Starting Celery Beat..."
	export DJANGO_SETTINGS_MODULE=config.settings; \
	poetry run celery -A config beat -l info & \
	@echo "🚀 Starting Django server..."
	export DJANGO_SETTINGS_MODULE=config.settings; \
	poetry run python manage.py runserver 0.0.0.0:8000 & \
	@echo "⚡ Triggering startup ETL task..."
	export DJANGO_SETTINGS_MODULE=config.settings; \
	poetry run python manage.py run_startup_etl

