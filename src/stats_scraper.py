import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import src.xg_parse_utils as pu


class StatsScraper:
    """
    This class communicates with NHL API and gathers stats.
    """
    def __init__(self, data_path):
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
        if season == 2012:      # Lockout season
            n_games = 720
        elif season < 2017:     # VGK came into NHL
            n_games = 1230
        else:
            n_games = 1271
        game_ids = [f"{season}02{i_game:04d}" for i_game in range(1, n_games+1)]
        if n_jobs == 1:   # Sequential download
            games = []
            for i_game in tqdm(game_ids):
                games.append(StatsScraper._get_game(i_game))
        else:   # Parallel download
            with mp.Pool(n_jobs) as p:
                games = p.map(StatsScraper._get_game, game_ids)
        # Save to a pickle
        file_path = self._data_path / "games_raw" / f"{season}-{season+1}.pickle"
        with open(file_path, "wb") as f:
            pickle.dump(games, f)
        end = time.time()
        print(f"\t... saved [{end - start:.2f} s] to '{file_path}'")

    @staticmethod
    def _get_game(game_id: str) -> dict:
        """
        Downloads one game from NHL API, filters it and returns as a json.

        :param game_id: str - game identifier in format <season>02<gameID> (e.g. 2015020001)
        :return: dict - json with lightweight info about the game (see _filter_game_json)
        """
        game_url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
        response = requests.get(game_url)
        if response.status_code != 200:
            response.raise_for_status()
        return StatsScraper._filter_game_json(response.json())

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
        i = 0
        for play in game['liveData']['plays']['allPlays']:
            if bool(play['coordinates']):
                play_dict = {
                    'id': i,
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
                i += 1
        filtered = StatsScraper.add_strength(game, filtered)
        return filtered

    @staticmethod
    def add_strength(game, filtered):
        """
        Scrapes strengths (5v5, PP, SH etc.) and empty nets from http://www.nhl.com/scores/htmlreports/ for each play in
        given game and adds them into the subresult 'filtered'.

        :param game:
        :param filtered:
        :return:
        """
        new_stats = []

        # Get a HTML report
        game_id = str(game['gamePk'])
        season = int(game_id[:4])
        url = f"http://www.nhl.com/scores/htmlreports/{season}{season + 1}/PL{game_id[4:]}.HTM"
        response = requests.get(url)
        if response.status_code != 200:
            response.raise_for_status()
        soup = BeautifulSoup(response.content, features="lxml")

        # Go through all events in the game
        trs = soup.find_all("tr", attrs={'class': ["evenColor", "oddColor"]})
        for tr in trs:
            columns = tr.find_all("td", recursive=False)
            away_on_ice = columns[6].find_all("td")  # players currently on ice
            home_on_ice = columns[7].find_all("td")
            play = {
                'id': columns[0].text,
                'type': columns[4].text,
                'strength': columns[2].text,
                'str_away': len(away_on_ice) // 4,
                'str_home': len(home_on_ice) // 4,
            }
            # check if empty net
            if away_on_ice and home_on_ice:
                play['empty_net_away'] = False if "G" in away_on_ice[-1].text else True
                play['empty_net_home'] = False if "G" in home_on_ice[-1].text else True
            new_stats.append(play)

        # Convert to pandas DataFrame
        ns = pd.DataFrame(new_stats)
        ns = ns[~ns['type'].isin(['PGSTR', 'PGEND', 'ANTHEM', 'PSTR', 'PEND', 'STOP', 'GEND'])]
        ns['str_away'] = pd.to_numeric(ns['str_away'], downcast="unsigned")
        ns['str_home'] = pd.to_numeric(ns['str_home'], downcast="unsigned")
        ns = ns.replace('\xa0', np.nan) \
            .drop('id', axis=1) \
            .reset_index(drop=True)

        # Merge strengths and empty nets with previous play info ('filtered')
        for play in filtered['plays']:
            ns_play = ns.loc[play['id']]
            if play['team']['id'] == filtered['teams']['home']['id']:
                play['strength_active'] = ns_play['str_home'].item()
                play['strength_opp'] = ns_play['str_away'].item()
                play['empty_net_opp'] = ns_play['empty_net_away']
            else:
                play['strength_active'] = ns_play['str_away'].item()
                play['strength_opp'] = ns_play['str_home'].item()
                play['empty_net_opp'] = ns_play['empty_net_home']
        return filtered

    def convert_season_to_xg_pandas(self, season, save_to_csv=False):
        """
        Convert jsons (pickle) into pandas DataFrame format for purpose of a xG model.

        :param season: int - season (e.g. 2015 for 2015/1016)
        :param save_to_csv: boolean - return the result or save it to a csv file
        :return:
        """
        print(f"## StatsExtractor: convert SEASON {season}/{season + 1} to a csv for a xG model.")
        start = time.time()
        shots = []

        with open(self._data_path / "games_raw" / f"{season}-{season+1}.pickle", "rb") as f:
            games = pickle.load(f)
        for game in games:
            shots.extend(pu.parse_game_for_xg(game))
        shots = pd.DataFrame(shots)
        if save_to_csv:
            end = time.time()
            filepath = self._data_path / "xg_pbp" / f"{season}-{season + 1}.csv"
            shots.to_csv(filepath)
            print(f"\t... saved [{end - start:.2f} s] to '{filepath}'")
        else:
            end = time.time()
            print(f"\t... done [{end - start:.2f} s]")
            return shots
