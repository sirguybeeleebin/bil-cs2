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


async def on_map_text(message: Message, state: FSMContext, api_base_url: str):
    data = await state.get_data()
    token = data["token"]
    query = message.text.strip()
    log.info(f"Пользователь {message.from_user.id} ищет карты с запросом: '{query}'")

    results = await search(api_base_url, token, "maps", query)
    if not results:
        await message.answer("❌ Карты не найдены.")
        log.warning(
            f"Пользователь {message.from_user.id}: карты не найдены для запроса '{query}'"
        )
        return

    kb = build_results_kb("maps", results)
    await message.answer("🗺 Выбери карту:", reply_markup=kb)
    log.info(f"Пользователю {message.from_user.id} отправлены результаты поиска карт")


async def on_map_selected(callback: CallbackQuery, state: FSMContext):
    _, map_id, map_name = callback.data.split(":", 2)
    await state.update_data(map_id=int(map_id), map_name=map_name)
    await callback.message.answer(
        f"✅ Карта выбрана: {map_name}\n\nТеперь введи часть названия первой команды:"
    )
    await state.set_state(PredictionStates.choosing_team1)
    await callback.answer()
    log.info(
        f"Пользователь {callback.from_user.id} выбрал карту '{map_name}' (ID: {map_id})"
    )
