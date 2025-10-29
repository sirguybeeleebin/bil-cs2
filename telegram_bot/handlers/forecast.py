import asyncio
import logging

from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from bot.api.client import forecast, forecast_result

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger(__name__)


async def confirm_data(
    message: Message, state: FSMContext, api_base_url: str, token: str
):
    data = await state.get_data()
    payload = {
        "map_id": data["map_id"],
        "team1_id": data["team1_id"],
        "team2_id": data["team2_id"],
        **{f"team1_player{i + 1}_id": data["players_team1"][i]["id"] for i in range(5)},
        **{f"team2_player{i + 1}_id": data["players_team2"][i]["id"] for i in range(5)},
    }

    log.info(
        f"Пользователь {message.from_user.id} отправляет данные для прогноза: {payload}"
    )
    await message.answer("⏳ Отправляю данные для прогноза...")

    forecast_response = await forecast(api_base_url, token, payload)
    if not forecast_response or "task_id" not in forecast_response:
        await message.answer("❌ Ошибка при отправке задачи прогнозирования.")
        log.warning(
            f"Ошибка при отправке прогноза для пользователя {message.from_user.id}"
        )
        return

    task_id = forecast_response["task_id"]
    await message.answer(
        f"✅ Задача создана, ожидаем результат (task_id: {task_id})..."
    )
    log.info(
        f"Пользователь {message.from_user.id} создал задачу прогнозирования с task_id: {task_id}"
    )

    for attempt in range(1, 31):
        result = await forecast_result(api_base_url, token, task_id)
        if result:
            t1_prob = result["team1_win_probability"]
            t2_prob = result["team2_win_probability"]
            text = f"📊 Прогноз:\n\n{data['team1_name']} — {t1_prob * 100:.1f}%\n{data['team2_name']} — {t2_prob * 100:.1f}%"
            await message.answer(text)
            log.info(
                f"Пользователь {message.from_user.id} получил результат прогноза: {text}"
            )
            return
        log.info(
            f"Пользователь {message.from_user.id}: результат прогноза ещё не готов, попытка {attempt}/30"
        )
        await asyncio.sleep(1.5)

    await message.answer("⌛ Прогноз не готов, попробуйте позже.")
    log.warning(
        f"Пользователь {message.from_user.id}: прогноз не готов после 30 попыток"
    )
