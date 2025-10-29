import logging

import httpx

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger(__name__)


async def get_token(base_url: str, username: str, password: str) -> str | None:
    url = f"{base_url}/auth/token/"
    payload = {"username": username, "password": password}
    log.info(f"Запрос токена для пользователя '{username}' на {url}")
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            log.warning(f"Ошибка получения токена: статус {resp.status_code}")
            return None
        token = resp.json().get("access")
        log.info("Токен успешно получен")
        return token


async def search(base_url: str, token: str, category: str, query: str) -> list[dict]:
    url = f"{base_url}/{category}/"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"name": query}
    log.info(f"Поиск '{query}' в категории '{category}' на {url}")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            log.warning(f"Ошибка поиска: статус {resp.status_code}")
            return []
        results = resp.json()
        log.info(f"Найдено {len(results)} результатов для запроса '{query}'")
        return results


async def forecast(base_url: str, token: str, payload: dict) -> dict | None:
    url = f"{base_url}/forecast/"
    headers = {"Authorization": f"Bearer {token}"}
    log.info(f"Отправка данных для прогноза на {url} с payload: {payload}")
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            log.warning(f"Ошибка создания прогноза: статус {resp.status_code}")
            return None
        result = resp.json()
        log.info(f"Прогноз успешно создан, task_id: {result.get('task_id')}")
        return result


async def forecast_result(base_url: str, token: str, task_id: str) -> dict | None:
    url = f"{base_url}/forecast/{task_id}/"
    headers = {"Authorization": f"Bearer {token}"}
    log.info(f"Запрос результата прогноза task_id={task_id} на {url}")
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 202:
            log.info(f"Результат прогноза task_id={task_id} еще не готов")
            return None
        if resp.status_code != 200:
            log.warning(
                f"Ошибка получения результата прогноза: статус {resp.status_code}"
            )
            return None
        result = resp.json()
        log.info(f"Результат прогноза task_id={task_id} получен")
        return result
