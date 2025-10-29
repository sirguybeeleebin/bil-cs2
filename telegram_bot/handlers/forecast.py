import asyncio

from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from facade import CS2ForecasterFacade, ForecastResultNotReady


async def confirm_data(
    message: Message, state: FSMContext, forecaster: CS2ForecasterFacade
):
    data = await state.get_data()

    payload = {
        "map_id": data["map_id"],
        "team1_id": data["team1_id"],
        "team2_id": data["team2_id"],
        **{f"team1_player{i + 1}_id": data["players_team1"][i]["id"] for i in range(5)},
        **{f"team2_player{i + 1}_id": data["players_team2"][i]["id"] for i in range(5)},
    }

    await message.answer("⏳ Отправляю данные для прогноза...")

    # --- Отправляем данные на /forecast/ ---
    forecast_response = await forecaster.submit_forecast(payload)
    if not forecast_response or "task_id" not in forecast_response:
        await message.answer("❌ Ошибка при отправке задачи прогнозирования.")
        return

    task_id = forecast_response["task_id"]
    await message.answer(
        f"✅ Задача создана, ожидаем результат (task_id: {task_id})..."
    )

    for _ in range(30):
        try:
            result = await forecaster.get_forecast_result(task_id)
        except ForecastResultNotReady:
            await asyncio.sleep(1.5)
            continue

        if result:
            t1_prob = result["team1_win_probability"]
            t2_prob = result["team2_win_probability"]
            result_text = (
                f"📊 Прогноз:\n\n"
                f"{data['team1_name']} — {t1_prob * 100:.1f}%\n"
                f"{data['team2_name']} — {t2_prob * 100:.1f}%"
            )
            await message.answer(result_text)
            return

    await message.answer("⌛ Прогноз не готов, попробуйте позже.")
