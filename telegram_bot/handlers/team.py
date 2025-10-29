import logging

from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from bot.api.client import search
from bot.keyboards.results import build_results_kb
from bot.states import PredictionStates

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger(__name__)


async def on_team_text(
    message: Message, state: FSMContext, api_base_url: str, team_number: int
):
    data = await state.get_data()
    token = data["token"]
    query = message.text.strip()
    log.info(
        f"Пользователь {message.from_user.id} ищет команду {team_number} с запросом: '{query}'"
    )

    results = await search(api_base_url, token, "teams", query)
    if not results:
        await message.answer("❌ Команды не найдены.")
        log.warning(
            f"Пользователь {message.from_user.id}: команды не найдены для запроса '{query}'"
        )
        return

    kb = build_results_kb(f"team{team_number}", results)
    text = (
        "🏆 Выбери первую команду:" if team_number == 1 else "⚔️ Выбери вторую команду:"
    )
    await message.answer(text, reply_markup=kb)
    log.info(
        f"Пользователю {message.from_user.id} отправлены результаты поиска команды {team_number}"
    )


async def on_team_selected(
    callback: CallbackQuery, state: FSMContext, team_number: int
):
    _, team_id, team_name = callback.data.split(":", 2)
    key_name = f"team{team_number}_id"
    key_display = f"team{team_number}_name"
    await state.update_data(**{key_name: int(team_id), key_display: team_name})

    log.info(
        f"Пользователь {callback.from_user.id} выбрал команду '{team_name}' (ID: {team_id}) для команды {team_number}"
    )

    if team_number == 1:
        await callback.message.answer("Теперь введи часть названия второй команды:")
        await state.set_state(PredictionStates.choosing_team2)
    else:
        await callback.message.answer(
            "👥 Теперь введи часть имени первого игрока команды 1:"
        )
        await state.update_data(players_team1=[])
        await state.set_state(PredictionStates.choosing_players_team1)
        log.info(
            f"Пользователь {callback.from_user.id} завершил выбор второй команды, начинаем набор игроков команды 1"
        )

    await callback.answer()
