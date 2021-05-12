import pickle

import pandas as pd
import requests
import time
import multiprocessing as mp
from pathlib import Path
import src.parsing_utils as pu


class StatsScraper:
    """
    This class communicates with NHL API and gathers stats.
    """
    def __init__(self, data_path):
        self._base_url = "https://statsapi.web.nhl.com/api/v1/"
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(f"'data_path' must be a string or instance of Path class. Not {type(data_path)}.")

    def download_season(self, season: int, n_jobs=4):
        """
        Downloads all regular season games from a given season, filters the jsons and saves list of jsons into a pickle.

        :param season: int - initial year of a season (e.g. 2015 for 2015/2016 season)
        :param n_jobs: int - number of parallel processes to use (default: 4)
        """
        print(f"## StatsExtractor: download SEASON {season}/{season+1} from NHL API")
        # Initialization
        start = time.time()
        if season == 2012:
            n_games = 720
        elif season < 2017:
            n_games = 1230
        else:
            n_games = 1271
        game_ids = [f"{season}02{i_game:04d}" for i_game in range(1, n_games+1)]
        # Parallel download
        with mp.Pool(n_jobs) as p:
            games = p.map(self._get_game, game_ids)
        # Save to a pickle
        file_path = self._data_path / "games_raw" / f"{season}-{season+1}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(games, f)
        end = time.time()
        print(f"... saved [{end - start:.2f} s] to '{file_path}'")

    def _get_game(self, game_id: str) -> dict:
        """
        Downloads one game from NHL API, filters it and returns as a json.

        :param game_id: str - game identifier in format <season>02<gameID> (e.g. 2015020001)
        :return: dict - json with lightweight info about the game (see _filter_game_json)
        """
        game_url = self._base_url + f"game/{game_id}/feed/live"
        response = requests.get(game_url)
        if response.status_code != 200:
            response.raise_for_status()
        return self._filter_game_json(response.json())

    @staticmethod
    def _filter_game_json(game: dict) -> dict:
        """
        TODO: update
        Filters given game dict (json). Keeps: id of the game, timestamp, teams (id, name, triCode), all plays with
        coordinates (type, team, coordinates). Plays with coordinates are mainly shots, penalty, goals, hits, giveaways,
        takeaways and face-offs.

        :param game: dict - raw json from NHL API
        :return: dict - filtered json
        """
        filtered = {'id': game['gamePk'],
                    'datetime': game['gameData']['datetime']['dateTime'],
                    'teams': {},
                    'plays': []}
        for team in ['home', 'away']:
            filtered['teams'][team] = {}
            for key in ['id', 'name', 'triCode']:
                filtered['teams'][team][key] = game['gameData']['teams'][team][key]
            filtered['teams'][team]['teamStats'] = game['liveData']['boxscore']['teams'][team]['teamStats']['teamSkaterStats']
        for play in game['liveData']['plays']['allPlays']:
            if bool(play['coordinates']):
                play_dict = {
                    'type': play['result']['eventTypeId'],
                    'team': play['team'],
                    'coordinates': play['coordinates'],
                    'score': play['about']['goals'],
                    'period': play['about']['period'],
                    'time': play['about']['periodTime']
                }
                if "secondaryType" in play['result']:
                    play_dict['shotType'] = play['result']['secondaryType']
                filtered['plays'].append(play_dict)
        filtered = pu.add_strength(game, filtered)
        return filtered

    def convert_season_to_xg_csv(self, season, to_csv=False):
        """
        Convert jsons (pkl) into csv format for xG model.

        :param filename: TODO
        :param season: TODO
        :return:
        """
        print(f"## StatsExtractor: convert SEASON {season}/{season + 1} to a csv for a xG model.")
        start = time.time()
        shots = []

        with open(self._data_path / "games_raw" / f"{season}-{season+1}.pkl", "rb") as f:
            games = pickle.load(f)
        for game in games:
            shots.extend(pu.parse_game(game))
        shots = pd.DataFrame(shots)
        if to_csv:
            end = time.time()
            filepath = self._data_path / "pbp_csv" / f"{season}-{season + 1}.csv"
            shots.to_csv(filepath)
            print(f"... saved [{end - start:.2f} s] to '{filepath}'")
        else:
            end = time.time()
            print(f"... done [{end - start:.2f} s]")
            return shots
