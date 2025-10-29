from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from facade import CS2ForecasterFacade
from handlers.forecast_handlers import confirm_data
from keyboards import build_results_kb
from states import PredictionStates


async def on_player1_text(
    message: Message, state: FSMContext, forecaster: CS2ForecasterFacade
):
    query = message.text.strip()
    results = await forecaster.search("players", query)

    if not results:
        await message.answer("❌ Игроков не найдено.")
        return

    kb = build_results_kb("player1", results)
    await message.answer("👤 Выбери игрока команды 1:", reply_markup=kb)


async def on_player1_selected(callback: CallbackQuery, state: FSMContext):
    _, player_id, player_name = callback.data.split(":", 2)
    data = await state.get_data()
    players = data.get("players_team1", [])
    players.append({"id": int(player_id), "name": player_name})
    await state.update_data(players_team1=players)

    if len(players) < 5:
        await callback.message.answer(
            f"✅ Добавлен {player_name} ({len(players)}/5). Следующий игрок:"
        )
    else:
        await callback.message.answer(
            "✅ Команда 1 собрана!\nТеперь вводим игроков команды 2:"
        )
        await state.update_data(players_team2=[])
        await state.set_state(PredictionStates.choosing_players_team2)

    await callback.answer()


async def on_player2_text(
    message: Message, state: FSMContext, forecaster: CS2ForecasterFacade
):
    query = message.text.strip()
    results = await forecaster.search("players", query)

    if not results:
        await message.answer("❌ Игроков не найдено.")
        return

    kb = build_results_kb("player2", results)
    await message.answer("👤 Выбери игрока команды 2:", reply_markup=kb)


async def on_player2_selected(
    callback: CallbackQuery, state: FSMContext, forecaster: CS2ForecasterFacade
):
    _, player_id, player_name = callback.data.split(":", 2)
    data = await state.get_data()
    players = data.get("players_team2", [])
    players.append({"id": int(player_id), "name": player_name})
    await state.update_data(players_team2=players)

    if len(players) < 5:
        await callback.message.answer(
            f"✅ Добавлен {player_name} ({len(players)}/5). Следующий игрок:"
        )
    else:
        await state.set_state(PredictionStates.confirming)
        await confirm_data(callback.message, state, forecaster)

    await callback.answer()
