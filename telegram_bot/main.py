import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.fsm.storage.memory import MemoryStorage

from telegram_bot.config import get_config
from telegram_bot.handlers.auth import password_handler, start, username_handler
from telegram_bot.handlers.maps import on_map_selected, on_map_text
from telegram_bot.handlers.players import on_player_selected, on_player_text
from telegram_bot.handlers.teams import on_team_selected, on_team_text

# ------------------------------
# Настройка логирования
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------
# Основная функция бота
# ------------------------------
async def main():
    log.info("Загрузка конфигурации...")
    config = get_config()

    bot = Bot(token=config["BOT_TOKEN"])
    dp = Dispatcher(storage=MemoryStorage())

    log.info("Регистрация обработчиков...")

    # --- Auth ---
    dp.message.register(start, F.text == "/start")
    dp.message.register(
        username_handler, state=F.state == "PredictionStates:waiting_username"
    )
    dp.message.register(
        lambda msg, state: password_handler(msg, state, config["API_BASE_URL"]),
        state=F.state == "PredictionStates:waiting_password",
    )

    # --- Map ---
    dp.message.register(
        lambda msg, state: on_map_text(msg, state, config["API_BASE_URL"]),
        state=F.state == "PredictionStates:choosing_map",
    )
    dp.callback_query.register(on_map_selected, F.data.startswith("maps:"))

    # --- Teams ---
    dp.message.register(
        lambda msg, state: on_team_text(msg, state, config["API_BASE_URL"], 1),
        state=F.state == "PredictionStates:choosing_team1",
    )
    dp.callback_query.register(
        lambda cb, state: on_team_selected(cb, state, 1), F.data.startswith("team1:")
    )
    dp.message.register(
        lambda msg, state: on_team_text(msg, state, config["API_BASE_URL"], 2),
        state=F.state == "PredictionStates:choosing_team2",
    )
    dp.callback_query.register(
        lambda cb, state: on_team_selected(cb, state, 2), F.data.startswith("team2:")
    )

    # --- Players ---
    dp.message.register(
        lambda msg, state: on_player_text(msg, state, config["API_BASE_URL"], 1),
        state=F.state == "PredictionStates:choosing_players_team1",
    )
    dp.callback_query.register(
        lambda cb, state: on_player_selected(cb, state, config["API_BASE_URL"], 1),
        F.data.startswith("player1:"),
    )
    dp.message.register(
        lambda msg, state: on_player_text(msg, state, config["API_BASE_URL"], 2),
        state=F.state == "PredictionStates:choosing_players_team2",
    )
    dp.callback_query.register(
        lambda cb, state: on_player_selected(cb, state, config["API_BASE_URL"], 2),
        F.data.startswith("player2:"),
    )

    log.info("🤖 Бот запущен и готов к работе.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.exception(f"Произошла ошибка при запуске бота: {e}")
