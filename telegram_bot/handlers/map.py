from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from facade import CS2ForecasterFacade
from keyboards import build_results_kb
from states import PredictionStates


async def start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("👋 Привет! Напиши часть названия карты:")
    await state.set_state(PredictionStates.choosing_map)


async def on_map_text(
    message: Message, state: FSMContext, forecaster: CS2ForecasterFacade
):
    query = message.text.strip()
    results = await forecaster.search("maps", query)
    if not results:
        await message.answer("❌ Карты не найдены.")
        return
    kb = build_results_kb("maps", results)
    await message.answer("🗺 Выбери карту:", reply_markup=kb)


async def on_map_selected(callback: CallbackQuery, state: FSMContext):
    _, map_id, map_name = callback.data.split(":", 2)
    await state.update_data(map_id=int(map_id), map_name=map_name)
    await callback.message.answer(
        f"✅ Карта выбрана: {map_name}\n\nТеперь введи часть названия первой команды:"
    )
    await state.set_state(PredictionStates.choosing_team1)
    await callback.answer()
