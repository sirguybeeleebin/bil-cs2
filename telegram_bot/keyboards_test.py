# from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
# from your_module import (
#     build_results_kb,
# )  # Replace 'your_module' with the actual module name


# def test_build_results_kb_empty():
#     """Test that an empty list returns an empty InlineKeyboardMarkup"""
#     result = build_results_kb("category", [])
#     assert isinstance(result, InlineKeyboardMarkup)
#     assert result.inline_keyboard == []


# def test_build_results_kb_single_item():
#     """Test that a single result produces one button with correct callback_data"""
#     results = [{"id": 1, "name": "Item1"}]
#     result = build_results_kb("cat", results)

#     assert isinstance(result, InlineKeyboardMarkup)
#     assert len(result.inline_keyboard) == 1
#     button = result.inline_keyboard[0][0]
#     assert isinstance(button, InlineKeyboardButton)
#     assert button.text == "Item1"
#     assert button.callback_data == "cat:1:Item1"


# def test_build_results_kb_multiple_items():
#     """Test that multiple results produce multiple buttons correctly"""
#     results = [
#         {"id": 1, "name": "Item1"},
#         {"id": 2, "name": "Item2"},
#         {"id": 3, "name": "Item3"},
#     ]
#     result = build_results_kb("cat", results)

#     assert isinstance(result, InlineKeyboardMarkup)
#     assert len(result.inline_keyboard) == 3

#     for i, item in enumerate(results):
#         button = result.inline_keyboard[i][0]
#         assert button.text == item["name"]
#         assert button.callback_data == f"cat:{item['id']}:{item['name']}"


# def test_build_results_kb_special_characters():
#     """Test that names with special characters are correctly handled"""
#     results = [{"id": 1, "name": "Item & Special"}]
#     result = build_results_kb("cat", results)

#     button = result.inline_keyboard[0][0]
#     assert button.text == "Item & Special"
#     assert button.callback_data == "cat:1:Item & Special"
