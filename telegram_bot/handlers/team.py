from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from facade import CS2ForecasterFacade
from keyboards import build_results_kb
from states import PredictionStates


async def on_team1_text(
    message: Message, state: FSMContext, forecaster: CS2ForecasterFacade
):
    query = message.text.strip()
    results = await forecaster.search("teams", query)

    if not results:
        await message.answer("❌ Команды не найдены.")
        return

    kb = build_results_kb("team1", results)
    await message.answer("🏆 Выбери первую команду:", reply_markup=kb)


async def on_team1_selected(callback: CallbackQuery, state: FSMContext):
    _, team_id, team_name = callback.data.split(":", 2)
    await state.update_data(team1_id=int(team_id), team1_name=team_name)
    await callback.message.answer(
        f"✅ Команда 1 выбрана: {team_name}\n\nТеперь введи часть названия второй команды:"
    )
    await state.set_state(PredictionStates.choosing_team2)
    await callback.answer()


async def on_team2_text(
    message: Message, state: FSMContext, forecaster: CS2ForecasterFacade
):
    query = message.text.strip()
    results = await forecaster.search("teams", query)

    if not results:
        await message.answer("❌ Команды не найдены.")
        return

    kb = build_results_kb("team2", results)
    await message.answer("⚔️ Выбери вторую команду:", reply_markup=kb)


async def on_team2_selected(callback: CallbackQuery, state: FSMContext):
    _, team_id, team_name = callback.data.split(":", 2)
    await state.update_data(team2_id=int(team_id), team2_name=team_name)
    await callback.message.answer(
        "👥 Теперь введи часть имени первого игрока команды 1:"
    )
    await state.update_data(players_team1=[])
    await state.set_state(PredictionStates.choosing_players_team1)
    await callback.answer()
