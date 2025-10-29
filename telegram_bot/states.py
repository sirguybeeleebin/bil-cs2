from aiogram.fsm.state import State, StatesGroup


class PredictionStates(StatesGroup):
    waiting_username = State()
    waiting_password = State()
    choosing_map = State()
    choosing_team1 = State()
    choosing_team2 = State()
    choosing_players_team1 = State()
    choosing_players_team2 = State()
    confirming = State()
