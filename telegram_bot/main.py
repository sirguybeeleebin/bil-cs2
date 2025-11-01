import asyncio
import logging
import os
from functools import partial
from typing import List

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")


# ---------------------- Models ----------------------


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MapResponse(BaseModel):
    map_id: int
    name: str


class TeamResponse(BaseModel):
    team_id: int
    name: str


class PlayerResponse(BaseModel):
    player_id: int
    name: str


class PlayersSelection(BaseModel):
    players: List[PlayerResponse] = []


class ForecastRequest(BaseModel):
    map_id: int
    team1_id: int
    team2_id: int
    team1_player1_id: int
    team1_player2_id: int
    team1_player3_id: int
    team1_player4_id: int
    team1_player5_id: int
    team2_player1_id: int
    team2_player2_id: int
    team2_player3_id: int
    team2_player4_id: int
    team2_player5_id: int


class ForecastResponse(BaseModel):
    team1_id: int
    team2_id: int
    team1_win_probability: float
    team2_win_probability: float


class MetricsResponse(BaseModel):
    train_result_id: str
    auc: float | None
    f1: float | None
    precision: float | None
    recall: float | None
    accuracy: float | None
    tp: int | None
    tn: int | None
    fp: int | None
    fn: int | None
    created_at: str


async def api_get_metrics(client: httpx.AsyncClient, token: str) -> MetricsResponse:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get(f"{API_BASE_URL}/metrics/", headers=headers)
    resp.raise_for_status()
    data = await resp.json()
    return MetricsResponse(**data)


class PredictionStates(StatesGroup):
    waiting_username = State()
    waiting_password = State()
    choosing_map = State()
    choosing_team1 = State()
    choosing_team2 = State()
    choosing_players_team1 = State()
    choosing_players_team2 = State()
    confirming = State()


# ---------------------- Keyboards ----------------------


def build_results_kb(prefix: str, items: list[BaseModel]):
    kb = InlineKeyboardBuilder()
    for item in items:
        id_value = (
            getattr(item, "map_id", None)
            or getattr(item, "team_id", None)
            or getattr(item, "player_id", None)
        )
        kb.button(text=item.name, callback_data=f"{prefix}:{id_value}:{item.name}")
    kb.adjust(1)
    return kb.as_markup()


def build_confirm_kb():
    kb = InlineKeyboardBuilder()
    kb.button(text="Получить прогноз", callback_data="forecast:confirm")
    kb.adjust(1)
    return kb.as_markup()


# ---------------------- API Functions ----------------------


async def api_register(client: httpx.AsyncClient, auth: AuthRequest) -> str:
    resp = await client.post(f"{API_BASE_URL}/register/", json=auth.dict())
    resp.raise_for_status()
    data = await resp.json()
    return data["access_token"]


async def api_login(client: httpx.AsyncClient, auth: AuthRequest) -> str:
    resp = await client.post(
        f"{API_BASE_URL}/token/",
        data={"username": auth.username, "password": auth.password},
    )
    resp.raise_for_status()
    data = await resp.json()
    return data["access"]


async def api_search_maps(
    client: httpx.AsyncClient, token: str, name: str
) -> List[MapResponse]:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get(
        f"{API_BASE_URL}/maps/",
        params={"name": name, "page": 1, "page_size": 5},
        headers=headers,
    )
    resp.raise_for_status()
    data = await resp.json()
    return [MapResponse(**m) for m in data]


async def api_search_teams(
    client: httpx.AsyncClient, token: str, name: str
) -> List[TeamResponse]:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get(
        f"{API_BASE_URL}/teams/",
        params={"name": name, "page": 1, "page_size": 5},
        headers=headers,
    )
    resp.raise_for_status()
    data = await resp.json()
    return [TeamResponse(**t) for t in data]


async def api_search_players(
    client: httpx.AsyncClient, token: str, name: str
) -> List[PlayerResponse]:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get(
        f"{API_BASE_URL}/players/",
        params={"name": name, "page": 1, "page_size": 5},
        headers=headers,
    )
    resp.raise_for_status()
    data = await resp.json()
    return [PlayerResponse(**p) for p in data]


async def api_create_forecast(
    client: httpx.AsyncClient, token: str, forecast: ForecastRequest
) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.post(
        f"{API_BASE_URL}/forecast/", json=forecast.dict(), headers=headers
    )
    resp.raise_for_status()
    data = await resp.json()
    return data["forecast_id"]


async def api_get_forecast_result(
    client: httpx.AsyncClient, token: str, forecast_id: str
) -> ForecastResponse:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get(f"{API_BASE_URL}/forecast/{forecast_id}/", headers=headers)
    resp.raise_for_status()
    data = await resp.json()
    return ForecastResponse(**data)


# ---------------------- Handlers ----------------------


async def register(message: Message, state: FSMContext, client: httpx.AsyncClient):
    await state.update_data(action="register")
    await message.answer("Введите username для регистрации:")
    await state.set_state(PredictionStates.waiting_username)


async def login(message: Message, state: FSMContext, client: httpx.AsyncClient):
    await state.update_data(action="login")
    await message.answer("Введите username для входа:")
    await state.set_state(PredictionStates.waiting_username)


async def username_handler(
    message: Message, state: FSMContext, client: httpx.AsyncClient
):
    await state.update_data(username=message.text.strip())
    await message.answer("Введите password:")
    await state.set_state(PredictionStates.waiting_password)


async def password_handler(
    message: Message, state: FSMContext, client: httpx.AsyncClient
):
    data = await state.get_data()
    auth = AuthRequest(username=data["username"], password=message.text.strip())
    action = data.get("action", "login")
    try:
        if action == "register":
            token = await api_register(client, auth)
        else:
            token = await api_login(client, auth)
        await state.update_data(token=token, username=auth.username)
        await message.answer("✅ Успешно!")
        await message.answer("Введите название карты для прогноза:")
        await state.set_state(PredictionStates.choosing_map)
    except Exception:
        await message.answer("❌ Ошибка авторизации или регистрации. Попробуйте снова.")
        await state.set_state(PredictionStates.waiting_username)


async def start(message: Message, state: FSMContext, client: httpx.AsyncClient):
    data = await state.get_data()
    token = data.get("token")
    username = data.get("username")
    if token and username:
        await message.answer(
            f"👋 Привет, {username}! Введите название карты для прогноза:"
        )
        await state.set_state(PredictionStates.choosing_map)
    else:
        await message.answer(
            "👋 Вы не вошли в систему. Используйте /login или /register."
        )


async def on_map_text(message: Message, state: FSMContext, client: httpx.AsyncClient):
    data = await state.get_data()
    maps = await api_search_maps(client, data["token"], message.text.strip())
    if not maps:
        await message.answer("❌ Карты не найдены.")
        return
    await message.answer(
        "🗺 Выберите карту:", reply_markup=build_results_kb("maps", maps)
    )


async def on_map_selected(
    callback: CallbackQuery, state: FSMContext, client: httpx.AsyncClient
):
    _, map_id, map_name = callback.data.split(":", 2)
    await state.update_data(map=MapResponse(map_id=int(map_id), name=map_name))
    await callback.message.answer("✅ Карта выбрана! Введите название первой команды:")
    await state.set_state(PredictionStates.choosing_team1)
    await callback.answer()


async def on_team_text(
    message: Message, state: FSMContext, client: httpx.AsyncClient, team_number: int
):
    data = await state.get_data()
    teams = await api_search_teams(client, data["token"], message.text.strip())
    if not teams:
        await message.answer("❌ Команды не найдены.")
        return
    await message.answer(
        f"🏆 Выберите команду {team_number}:",
        reply_markup=build_results_kb(f"team{team_number}", teams),
    )


async def on_team_selected(
    callback: CallbackQuery,
    state: FSMContext,
    client: httpx.AsyncClient,
    team_number: int,
):
    _, team_id, team_name = callback.data.split(":", 2)
    team = TeamResponse(team_id=int(team_id), name=team_name)
    await state.update_data(**{f"team{team_number}": team})
    if team_number == 1:
        await callback.message.answer("Введите название второй команды:")
        await state.set_state(PredictionStates.choosing_team2)
    else:
        await callback.message.answer("Введите имя первого игрока команды 1:")
        await state.update_data(players_team1=PlayersSelection(players=[]))
        await state.set_state(PredictionStates.choosing_players_team1)
    await callback.answer()


async def on_player_text(
    message: Message, state: FSMContext, client: httpx.AsyncClient, team_number: int
):
    data = await state.get_data()
    players = await api_search_players(client, data["token"], message.text.strip())
    if not players:
        await message.answer("❌ Игроки не найдены.")
        return
    await message.answer(
        f"👤 Выберите игрока команды {team_number}:",
        reply_markup=build_results_kb(f"player{team_number}", players),
    )


async def on_player_selected(
    callback: CallbackQuery,
    state: FSMContext,
    client: httpx.AsyncClient,
    team_number: int,
):
    data = await state.get_data()
    _, player_id, player_name = callback.data.split(":", 2)
    player = PlayerResponse(player_id=int(player_id), name=player_name)

    key = f"players_team{team_number}"
    players_selection: PlayersSelection = data.get(key, PlayersSelection(players=[]))
    players_selection.players.append(player)
    await state.update_data(**{key: players_selection})

    if len(players_selection.players) < 5:
        await callback.message.answer(
            f"✅ Игрок {player_name} добавлен ({len(players_selection.players)}/5). Следующий игрок:"
        )
    else:
        if team_number == 1:
            await callback.message.answer(
                "✅ Команда 1 собрана! Теперь выбираем игроков команды 2:"
            )
            await state.update_data(players_team2=PlayersSelection(players=[]))
            await state.set_state(PredictionStates.choosing_players_team2)
        else:
            await state.set_state(PredictionStates.confirming)
            await callback.message.answer(
                "Все игроки выбраны! Нажмите кнопку для прогноза:",
                reply_markup=build_confirm_kb(),
            )
    await callback.answer()


async def on_forecast(
    callback: CallbackQuery, state: FSMContext, client: httpx.AsyncClient
):
    data = await state.get_data()
    forecast_request = ForecastRequest(
        map_id=data["map"].map_id,
        team1_id=data["team1"].team_id,
        team2_id=data["team2"].team_id,
        team1_player1_id=data["players_team1"].players[0].player_id,
        team1_player2_id=data["players_team1"].players[1].player_id,
        team1_player3_id=data["players_team1"].players[2].player_id,
        team1_player4_id=data["players_team1"].players[3].player_id,
        team1_player5_id=data["players_team1"].players[4].player_id,
        team2_player1_id=data["players_team2"].players[0].player_id,
        team2_player2_id=data["players_team2"].players[1].player_id,
        team2_player3_id=data["players_team2"].players[2].player_id,
        team2_player4_id=data["players_team2"].players[3].player_id,
        team2_player5_id=data["players_team2"].players[4].player_id,
    )

    forecast_id = await api_create_forecast(client, data["token"], forecast_request)
    await callback.message.answer("⏳ Прогноз создается, ожидайте...")

    forecast_result = await api_get_forecast_result(client, data["token"], forecast_id)

    text = (
        f"📊 Прогноз:\n"
        f"{data['team1'].name} — {forecast_result.team1_win_probability * 100:.1f}%\n"
        f"{data['team2'].name} — {forecast_result.team2_win_probability * 100:.1f}%"
    )
    await callback.message.answer(text)
    await state.clear()
    await state.set_state(PredictionStates.choosing_map)
    await callback.answer("✅ Вы можете ввести новую карту для следующего прогноза!")


async def metrics(message: Message, state: FSMContext, client: httpx.AsyncClient):
    data = await state.get_data()
    token = data.get("token")

    if not token:
        await message.answer("❌ Вы не вошли в систему. Используйте /login.")
        return

    try:
        metrics = await api_get_metrics(client, token)
        text = (
            f"📊 *Последние метрики модели:*\n"
            f"🆔 `{metrics.train_result_id}`\n"
            f"📅 {metrics.created_at}\n\n"
            f"🎯 Accuracy: {metrics.accuracy:.3f}\n"
            f"🏹 Precision: {metrics.precision:.3f}\n"
            f"💡 Recall: {metrics.recall:.3f}\n"
            f"⚖️ F1 Score: {metrics.f1:.3f}\n"
            f"📈 AUC: {metrics.auc:.3f}\n\n"
            f"✅ TP: {metrics.tp}\n"
            f"❌ FP: {metrics.fp}\n"
            f"🟢 TN: {metrics.tn}\n"
            f"🔴 FN: {metrics.fn}"
        )
        await message.answer(text, parse_mode="Markdown")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            await message.answer("❌ Метрики не найдены.")
        else:
            await message.answer("⚠️ Ошибка при получении метрик.")
    except Exception as e:
        log.exception(e)
        await message.answer("⚠️ Произошла ошибка при запросе метрик.")


async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())

    async with httpx.AsyncClient() as client:
        dp.message.register(partial(register, client=client), F.text == "/register")
        dp.message.register(partial(login, client=client), F.text == "/login")
        dp.message.register(partial(start, client=client), F.text == "/start")
        dp.message.register(partial(metrics, client=client), F.text == "/metrics")

        dp.message.register(
            partial(username_handler, client=client),
            StateFilter(PredictionStates.waiting_username),
        )
        dp.message.register(
            partial(password_handler, client=client),
            StateFilter(PredictionStates.waiting_password),
        )

        dp.message.register(
            partial(on_map_text, client=client),
            StateFilter(PredictionStates.choosing_map),
        )
        dp.callback_query.register(
            partial(on_map_selected, client=client), F.data.startswith("maps:")
        )

        dp.message.register(
            partial(on_team_text, client=client, team_number=1),
            StateFilter(PredictionStates.choosing_team1),
        )
        dp.callback_query.register(
            partial(on_team_selected, client=client, team_number=1),
            F.data.startswith("team1:"),
        )
        dp.message.register(
            partial(on_team_text, client=client, team_number=2),
            StateFilter(PredictionStates.choosing_team2),
        )
        dp.callback_query.register(
            partial(on_team_selected, client=client, team_number=2),
            F.data.startswith("team2:"),
        )

        dp.message.register(
            partial(on_player_text, client=client, team_number=1),
            StateFilter(PredictionStates.choosing_players_team1),
        )
        dp.callback_query.register(
            partial(on_player_selected, client=client, team_number=1),
            F.data.startswith("player1:"),
        )
        dp.message.register(
            partial(on_player_text, client=client, team_number=2),
            StateFilter(PredictionStates.choosing_players_team2),
        )
        dp.callback_query.register(
            partial(on_player_selected, client=client, team_number=2),
            F.data.startswith("player2:"),
        )

        dp.callback_query.register(
            partial(on_forecast, client=client), F.data == "forecast:confirm"
        )

        log.info("🤖 Бот запущен и готов к работе.")
        await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.exception(f"Произошла ошибка при запуске бота: {e}")
