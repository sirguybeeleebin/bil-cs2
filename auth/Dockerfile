FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Устанавливаем зависимости для сборки и curl для healthcheck
RUN apt-get update && \
    apt-get install -y gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Копируем конфигурацию poetry
COPY pyproject.toml poetry.lock* /app/

# Устанавливаем poetry
RUN pip install --no-cache-dir poetry

# Устанавливаем зависимости проекта без создания виртуального окружения
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Копируем исходники
COPY auth /app/auth

EXPOSE 8000

# Запуск сервиса
CMD ["poetry", "run", "uvicorn", "auth.main:app", "--host", "0.0.0.0", "--port", "8000"]
