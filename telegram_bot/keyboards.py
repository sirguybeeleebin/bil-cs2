import logging

from aiogram.utils.keyboard import InlineKeyboardBuilder

log = logging.getLogger(__name__)


def build_results_kb(prefix: str, results: list[dict]):
    kb = InlineKeyboardBuilder()
    for item in results:
        kb.button(
            text=item["name"], callback_data=f"{prefix}:{item['id']}:{item['name']}"
        )
        log.info(
            f"Добавлена кнопка: '{item['name']}' с callback_data='{prefix}:{item['id']}:{item['name']}'"
        )

    kb.adjust(1)
    log.info(
        f"Клавиатура с префиксом '{prefix}' построена, всего кнопок: {len(results)}"
    )
    return kb.as_markup()
