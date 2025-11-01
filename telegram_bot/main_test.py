from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from telegram_bot.main import (
    AuthRequest,
    ForecastRequest,
    ForecastResponse,
    PredictionStates,
    api_create_forecast,
    api_get_forecast_result,
    api_login,
    api_register,
    api_search_maps,
    api_search_players,
    api_search_teams,
    login,
    on_forecast,
    on_map_text,
    password_handler,
    register,
    start,
    username_handler,
)

# -------------------- API tests --------------------


@pytest.mark.asyncio
async def test_api_register():
    client = AsyncMock()
    client.post.return_value.raise_for_status = AsyncMock()
    client.post.return_value.json = AsyncMock(
        return_value={"access_token": "fake_token"}
    )

    auth = AuthRequest(username="user", password="pass")
    token = await api_register(client, auth)
    assert token == "fake_token"


@pytest.mark.asyncio
async def test_api_login():
    client = AsyncMock()
    client.post.return_value.raise_for_status = AsyncMock()
    client.post.return_value.json = AsyncMock(return_value={"access": "fake_token"})

    auth = AuthRequest(username="user", password="pass")
    token = await api_login(client, auth)
    assert token == "fake_token"


@pytest.mark.asyncio
async def test_api_search_maps():
    client = AsyncMock()
    client.get.return_value.raise_for_status = AsyncMock()
    client.get.return_value.json = AsyncMock(
        return_value=[{"map_id": 1, "name": "Dust2"}]
    )

    result = await api_search_maps(client, "token", "Dust2")
    assert len(result) == 1
    assert result[0].name == "Dust2"


@pytest.mark.asyncio
async def test_api_search_teams():
    client = AsyncMock()
    client.get.return_value.raise_for_status = AsyncMock()
    client.get.return_value.json = AsyncMock(
        return_value=[{"team_id": 1, "name": "TeamA"}]
    )

    result = await api_search_teams(client, "token", "TeamA")
    assert len(result) == 1
    assert result[0].name == "TeamA"


@pytest.mark.asyncio
async def test_api_search_players():
    client = AsyncMock()
    client.get.return_value.raise_for_status = AsyncMock()
    client.get.return_value.json = AsyncMock(
        return_value=[{"player_id": 1, "name": "Player1"}]
    )

    result = await api_search_players(client, "token", "Player1")
    assert len(result) == 1
    assert result[0].name == "Player1"


@pytest.mark.asyncio
async def test_api_create_forecast_and_result():
    client = AsyncMock()
    client.post.return_value.raise_for_status = AsyncMock()
    client.post.return_value.json = AsyncMock(return_value={"forecast_id": "123"})
    client.get.return_value.raise_for_status = AsyncMock()
    client.get.return_value.json = AsyncMock(
        return_value={
            "team1_id": 1,
            "team2_id": 2,
            "team1_win_probability": 0.6,
            "team2_win_probability": 0.4,
        }
    )

    forecast_req = ForecastRequest(
        map_id=1,
        team1_id=1,
        team2_id=2,
        team1_player1_id=1,
        team1_player2_id=2,
        team1_player3_id=3,
        team1_player4_id=4,
        team1_player5_id=5,
        team2_player1_id=6,
        team2_player2_id=7,
        team2_player3_id=8,
        team2_player4_id=9,
        team2_player5_id=10,
    )

    forecast_id = await api_create_forecast(client, "token", forecast_req)
    assert forecast_id == "123"

    result = await api_get_forecast_result(client, "token", forecast_id)
    assert result.team1_win_probability == 0.6
    assert result.team2_win_probability == 0.4


# -------------------- FSM handler tests --------------------


@pytest.mark.asyncio
async def test_register_handler():
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    state = AsyncMock(spec=FSMContext)
    client = AsyncMock()

    await register(message, state, client)
    message.answer.assert_called_once()
    state.set_state.assert_called_with(PredictionStates.waiting_username)


@pytest.mark.asyncio
async def test_login_handler():
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    state = AsyncMock(spec=FSMContext)
    client = AsyncMock()

    await login(message, state, client)
    message.answer.assert_called_once()
    state.set_state.assert_called_with(PredictionStates.waiting_username)


@pytest.mark.asyncio
async def test_username_handler():
    message = AsyncMock(spec=Message)
    message.text = "myuser"
    message.answer = AsyncMock()
    state = AsyncMock(spec=FSMContext)
    state.get_data.return_value = {}
    client = AsyncMock()

    await username_handler(message, state, client)
    state.update_data.assert_called()
    message.answer.assert_called()


@pytest.mark.asyncio
async def test_password_handler_login():
    message = AsyncMock(spec=Message)
    message.text = "mypass"
    message.answer = AsyncMock()
    state = AsyncMock(spec=FSMContext)
    state.get_data.return_value = {"username": "user", "action": "login"}
    client = AsyncMock()
    client.post.return_value.raise_for_status = AsyncMock()
    client.post.return_value.json = AsyncMock(return_value={"access": "token"})

    await password_handler(message, state, client)
    state.update_data.assert_called()
    message.answer.assert_called()


@pytest.mark.asyncio
async def test_start_handler_logged_in():
    message = AsyncMock(spec=Message)
    message.answer = AsyncMock()
    state = AsyncMock(spec=FSMContext)
    state.get_data.return_value = {"username": "user", "token": "token"}
    client = AsyncMock()

    await start(message, state, client)
    message.answer.assert_called()
    state.set_state.assert_called_with(PredictionStates.choosing_map)


@pytest.mark.asyncio
async def test_on_map_text_found():
    message = AsyncMock(spec=Message)
    message.text = "Dust2"
    message.answer = AsyncMock()
    state = AsyncMock(spec=FSMContext)
    state.get_data.return_value = {"token": "token"}
    client = AsyncMock()
    client.get.return_value.raise_for_status = AsyncMock()
    client.get.return_value.json = AsyncMock(
        return_value=[{"map_id": 1, "name": "Dust2"}]
    )

    await on_map_text(message, state, client)
    message.answer.assert_called()


# -------------------- on_forecast tests (without ForecasterService) --------------------


@pytest.mark.asyncio
async def test_on_forecast_flow():
    # Ensure message.answer is awaitable
    message_mock = AsyncMock(spec=Message)
    message_mock.answer = AsyncMock()

    callback = AsyncMock(spec=CallbackQuery)
    callback.message = message_mock
    callback.data = "forecast:confirm"
    callback.answer = AsyncMock()

    state = AsyncMock(spec=FSMContext)
    state.get_data.return_value = {
        "token": "fake-token",
        "map": MagicMock(map_id=1, name="Dust2"),
        "team1": MagicMock(team_id=1, name="TeamA"),
        "team2": MagicMock(team_id=2, name="TeamB"),
        "players_team1": MagicMock(
            players=[MagicMock(player_id=i) for i in range(1, 6)]
        ),
        "players_team2": MagicMock(
            players=[MagicMock(player_id=i) for i in range(6, 11)]
        ),
    }

    with (
        patch(
            "telegram_bot.main.api_create_forecast", new_callable=AsyncMock
        ) as mock_create,
        patch(
            "telegram_bot.main.api_get_forecast_result", new_callable=AsyncMock
        ) as mock_get,
    ):
        mock_create.return_value = "forecast-123"
        mock_get.return_value = ForecastResponse(
            team1_id=1, team2_id=2, team1_win_probability=0.6, team2_win_probability=0.4
        )

        await on_forecast(callback, state, client=AsyncMock())

        mock_create.assert_awaited_once()
        mock_get.assert_awaited_once()
        message_mock.answer.assert_awaited()
        callback.answer.assert_awaited()
        state.clear.assert_awaited()
        state.set_state.assert_awaited_with(PredictionStates.choosing_map)
