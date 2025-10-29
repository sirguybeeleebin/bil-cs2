from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def build_results_kb(category: str, results: list[dict]) -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(
                text=item["name"],
                callback_data=f"{category}:{item['id']}:{item['name']}",
            )
        ]
        for item in results
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
