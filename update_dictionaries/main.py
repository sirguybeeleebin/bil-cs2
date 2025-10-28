import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    load_dotenv()

    DATA_DIR = os.getenv("DATA_DIR", "data/games_raw")
    MAPS_DIR = os.getenv("MAPS_DIR", "data/maps")
    TEAMS_DIR = os.getenv("TEAMS_DIR", "data/teams")
    PLAYERS_DIR = os.getenv("PLAYERS_DIR", "data/players")

    if not os.path.isdir(DATA_DIR):
        log.error(f"Папка с данными не найдена: {DATA_DIR}")
        return

    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(TEAMS_DIR, exist_ok=True)
    os.makedirs(PLAYERS_DIR, exist_ok=True)

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"Ошибка чтения файла {filename}: {e}")
            continue

        now_iso = datetime.utcnow().isoformat()

        try:
            map_data = {"map_id": data["map"]["id"], "name": data["map"]["name"], "updated_at": now_iso}
            map_file = os.path.join(MAPS_DIR, f"{map_data['map_id']}.json")
            with open(map_file, "w", encoding="utf-8") as mf:
                json.dump(map_data, mf, ensure_ascii=False, indent=2)
            log.info(f"Сохранили карту: {map_data['name']} (ID: {map_data['map_id']})")
        except Exception:
            pass

        try:
            for p in data["players"]:
                try:
                    team_data = {
                        "team_id": p["team"]["id"],
                        "name": p["team"]["name"],
                        "updated_at": now_iso
                    }
                    team_file = os.path.join(TEAMS_DIR, f"{team_data['team_id']}.json")
                    with open(team_file, "w", encoding="utf-8") as tf:
                        json.dump(team_data, tf, ensure_ascii=False, indent=2)
                    log.info(f"Сохранили команду: {team_data['name']} (ID: {team_data['team_id']})")
                except Exception as e:
                    log.error(f"Ошибка при сохранении команды: {e}")
        except:
            continue

        try:
            for p in data["players"]:
                try:
                    player_data = {
                        "player_id": p["player"]["id"],
                        "name": p["player"]["name"],
                        "updated_at": now_iso
                    }
                    player_file = os.path.join(PLAYERS_DIR, f"{player_data['player_id']}.json")
                    with open(player_file, "w", encoding="utf-8") as pf:
                        json.dump(player_data, pf, ensure_ascii=False, indent=2)
                    log.info(f"Сохранили игрока: {player_data['name']} (ID: {player_data['player_id']})")
                except Exception as e:
                    log.error(f"Ошибка при сохранении игрока: {e}")
        except:
            continue


if __name__ == "__main__":
    main()
