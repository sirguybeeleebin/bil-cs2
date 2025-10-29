import asyncio

from aiogram import Bot, Dispatcher, F

from telegram_bot.config import API_BASE_URL, BOT_TOKEN
from telegram_bot.cs2_api_client import CS2ApiClient
from telegram_bot.handlers import map_handlers, player_handlers, team_handlers
from telegram_bot.states import PredictionStates


async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    # --- Инициализация клиента для работы с API ---
    api_client = CS2ApiClient(base_url=API_BASE_URL)

    # --- Map Handlers ---
    dp.message.register(map_handlers.start, F.text == "/start")
    dp.message.register(
        map_handlers.on_map_text, PredictionStates.choosing_map, api_client=api_client
    )
    dp.callback_query.register(
        map_handlers.on_map_selected, F.data.startswith("maps:"), api_client=api_client
    )

    # --- Team Handlers ---
    dp.message.register(
        team_handlers.on_team1_text,
        PredictionStates.choosing_team1,
        api_client=api_client,
    )
    dp.callback_query.register(
        team_handlers.on_team1_selected,
        F.data.startswith("team1:"),
        api_client=api_client,
    )

    dp.message.register(
        team_handlers.on_team2_text,
        PredictionStates.choosing_team2,
        api_client=api_client,
    )
    dp.callback_query.register(
        team_handlers.on_team2_selected,
        F.data.startswith("team2:"),
        api_client=api_client,
    )

    # --- Player Handlers ---
    dp.message.register(
        player_handlers.on_player1_text,
        PredictionStates.choosing_players_team1,
        api_client=api_client,
    )
    dp.callback_query.register(
        player_handlers.on_player1_selected,
        F.data.startswith("player1:"),
        api_client=api_client,
    )

    dp.message.register(
        player_handlers.on_player2_text,
        PredictionStates.choosing_players_team2,
        api_client=api_client,
    )
    dp.callback_query.register(
        player_handlers.on_player2_selected,
        F.data.startswith("player2:"),
        api_client=api_client,
    )

    print("🤖 Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
