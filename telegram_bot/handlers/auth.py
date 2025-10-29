import logging

from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from bot.api.client import get_token
from bot.states import PredictionStates

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
log = logging.getLogger(__name__)


async def start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("👋 Привет! Введите username для авторизации:")
    await state.set_state(PredictionStates.waiting_username)
    log.info(f"Пользователь {message.from_user.id} начал процесс авторизации.")


async def username_handler(message: Message, state: FSMContext):
    username = message.text.strip()
    await state.update_data(username=username)
    await message.answer("Введите password:")
    await state.set_state(PredictionStates.waiting_password)
    log.info(f"Пользователь {message.from_user.id} ввел username: {username}")


async def password_handler(message: Message, state: FSMContext, api_base_url: str):
    data = await state.get_data()
    username = data.get("username")
    password = message.text.strip()

    log.info(
        f"Пользователь {message.from_user.id} пытается авторизоваться с username: {username}"
    )

    token = await get_token(api_base_url, username, password)
    if not token:
        await message.answer(
            "❌ Ошибка авторизации. Попробуйте снова. Введите username:"
        )
        await state.set_state(PredictionStates.waiting_username)
        log.warning(
            f"Неудачная авторизация пользователя {message.from_user.id} с username: {username}"
        )
        return

    await state.update_data(token=token)
    await message.answer("✅ Успешно авторизован! Введите часть названия карты:")
    await state.set_state(PredictionStates.choosing_map)
    log.info(
        f"Пользователь {message.from_user.id} успешно авторизован с username: {username}"
    )
