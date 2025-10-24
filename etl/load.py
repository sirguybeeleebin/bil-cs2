import logging

import httpx

log = logging.getLogger("etl_worker")


def load_map(
    map_data: dict, client: httpx.Client, url: str, headers: dict | None = None
) -> None:
    if map_data is None:
        return
    try:
        client.post(url, json=map_data, headers=headers).raise_for_status()
    except httpx.HTTPError as e:
        log.error(f"Не удалось загрузить карту {map_data.get('map_id')}: {e}")


def load_teams(
    teams_data: list[dict], client: httpx.Client, url: str, headers: dict | None = None
) -> None:
    if not teams_data:
        return
    for team in teams_data:
        try:
            client.post(url, json=team, headers=headers).raise_for_status()
        except httpx.HTTPError as e:
            log.error(f"Не удалось загрузить команду {team.get('team_id')}: {e}")


def load_players(
    players_data: list[dict],
    client: httpx.Client,
    url: str,
    headers: dict | None = None,
) -> None:
    if not players_data:
        return
    for player in players_data:
        try:
            client.post(url, json=player, headers=headers).raise_for_status()
        except httpx.HTTPError as e:
            log.error(f"Не удалось загрузить игрока {player.get('player_id')}: {e}")
