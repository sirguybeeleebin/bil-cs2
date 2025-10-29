import logging

from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from bot.api.client import search
from bot.handlers.forecast import confirm_data
from bot.keyboards.results import build_results_kb
from bot.states import PredictionStates

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger(__name__)


async def on_player_text(
    message: Message, state: FSMContext, api_base_url: str, team_number: int
):
    data = await state.get_data()
    token = data["token"]
    query = message.text.strip()
    log.info(
        f"Пользователь {message.from_user.id} ищет игроков команды {team_number} с запросом: '{query}'"
    )

    results = await search(api_base_url, token, "players", query)
    if not results:
        await message.answer("❌ Игроков не найдено.")
        log.warning(
            f"Пользователь {message.from_user.id}: игроки не найдены для запроса '{query}'"
        )
        return

    kb = build_results_kb(f"player{team_number}", results)
    await message.answer(f"👤 Выбери игрока команды {team_number}:", reply_markup=kb)
    log.info(
        f"Пользователю {message.from_user.id} отправлены результаты поиска игроков команды {team_number}"
    )


async def on_player_selected(
    callback: CallbackQuery, state: FSMContext, api_base_url: str, team_number: int
):
    data = await state.get_data()
    token = data["token"]
    _, player_id, player_name = callback.data.split(":", 2)
    key = f"players_team{team_number}"
    players = data.get(key, [])
    players.append({"id": int(player_id), "name": player_name})
    await state.update_data(**{key: players})

    log.info(
        f"Пользователь {callback.from_user.id} выбрал игрока '{player_name}' (ID: {player_id}) команды {team_number}"
    )

    if len(players) < 5:
        await callback.message.answer(
            f"✅ Добавлен {player_name} ({len(players)}/5). Следующий игрок:"
        )
    else:
        if team_number == 1:
            await callback.message.answer(
                "✅ Команда 1 собрана!\nТеперь вводим игроков команды 2:"
            )
            await state.update_data(players_team2=[])
            await state.set_state(PredictionStates.choosing_players_team2)
            log.info(f"Пользователь {callback.from_user.id} завершил выбор команды 1")
        else:
            await state.set_state(PredictionStates.confirming)
            log.info(
                f"Пользователь {callback.from_user.id} завершил выбор команды 2, начинаем прогноз"
            )
            await confirm_data(callback.message, state, api_base_url, token)

    await callback.answer()
