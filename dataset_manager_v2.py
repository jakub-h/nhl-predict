import pandas as pd
import numpy as np
import json
import requests
import time
import os
from collections import deque


class DatasetManager():
    def __init__(self,
                 base_url="https://statsapi.web.nhl.com/api/v1/",
                 player_names_fn="player_names_v2.json",
                 team_names_fn="team_names_v2.json",
                 base_dataset_fn="base_dataset_v2.csv"):
        print("## DatasetManager V2 ##: Initialization")
        # Base URL for NHL API
        self._base_url = base_url

        # Players dictionary (IDs to names and vice versa)
        if player_names_fn is not None and player_names_fn in os.listdir("./"):
            with open(player_names_fn, 'r') as f:
                self._player_names = json.load(f)
            print("--> players 'ID <-> name' dictionary loaded from: {}".format(player_names_fn))
        else:
            self._player_names = None

        # Teams dictionary (IDs to short names and vice versa)
        if team_names_fn is not None and team_names_fn in os.listdir("./"):
            with open(team_names_fn, 'r') as f:
                self._team_names = json.load(f)
            print("--> teams 'ID <-> name' dictionary loaded from: {}".format(team_names_fn))
        else:
            self._team_names = None

        # Base dataset (full dataset, with team names, not standardized etc.)
        if base_dataset_fn is not None and base_dataset_fn in os.listdir("./"):
            self._base_dataset = pd.read_csv(base_dataset_fn, header=0, index_col=0)
            print("--> Base dataset loaded from: {}".format(base_dataset_fn))
        else:
            self._base_dataset = None

        self._coach_encoder = None
        self._scaler = None
    

    def _seek_game(self, season_year, game_num, preseason=False):
        game_type = "02"
        if preseason:
            game_type = "01"
        game_url = self._base_url + "game/{}{}{:04d}/feed/live".format(season_year, game_type, int(game_num))
        response = requests.get(game_url)
        return response.json()

    
    def _parse_game(self, game_dict):
        parsed_game = {}
        if 'gamePk' not in game_dict:
            return None
        if game_dict['gameData']['status']['abstractGameState'] != "Final":
            return None
        parsed_game['game_id'] = game_dict['gamePk']
        timestamp = time.strptime(game_dict['gameData']['datetime']['dateTime'], "%Y-%m-%dT%H:%M:%SZ")
        parsed_game['year'] = timestamp.tm_year
        parsed_game['month'] = timestamp.tm_mon
        parsed_game['day'] = timestamp.tm_mday
        parsed_game['hour'] = timestamp.tm_hour
        parsed_game['reg_draw'] = int(game_dict['liveData']['linescore']['currentPeriod'] > 3)
        for h_a in ['home', 'away']:
            parsed_game['{}_team_id'.format(h_a)] = game_dict['gameData']['teams'][h_a]['id']
            parsed_game['{}_team'.format(h_a)] = game_dict['gameData']['teams'][h_a]['triCode']
            parsed_game['{}_goals'.format(h_a)] = game_dict['liveData']['linescore']['teams'][h_a]['goals']
            parsed_game['{}_coach'.format(h_a)] = game_dict['liveData']['boxscore']['teams'][h_a]['coaches'][0]['person']['fullName']
        parsed_game['players'] = {'home': {}, 'away': {}}
        for h_a in ['home', 'away']:
            for player_type in ['goalie', 'skater']:
                for player_id in game_dict['liveData']['boxscore']['teams'][h_a]['{}s'.format(player_type)]:
                    if player_id not in game_dict['liveData']['boxscore']['teams'][h_a]['scratches']:
                        parsed_game['players'][h_a][player_id] = {}
                        raw_icetime = game_dict['liveData']['boxscore']['teams'][h_a]['players']['ID{}'.format(player_id)]['stats']['{}Stats'.format(player_type)]['timeOnIce']
                        minutes, seconds = raw_icetime.split(":")
                        parsed_game['players'][h_a][player_id]['icetime'] = 60*int(minutes) + int(seconds)
                        parsed_game['players'][h_a][player_id]['position'] = game_dict['liveData']['boxscore']['teams'][h_a]['players']['ID{}'.format(player_id)]['position']['code']
        return parsed_game


    def _game_to_json(self, season_year, game_num, preseason=False):
        game = self._seek_game(season_year, game_num, preseason)
        game = self._parse_game(game)
        if game is None:
            return
        game_type = "02"
        if preseason:
            game_type = "01"
        with open('games/v2/{}/{}{:04d}.json'.format(season_year, game_type, game_num), 'w', encoding='utf-8') as f:
            json.dump(game, f, ensure_ascii=False, indent=4)


    def get_season(self, season_year, preseason=False):
        print("## DatasetManager V2 ##: Downloading season {}/{} (preseason = {})".format(season_year, season_year+1, preseason))
        print("[", end=" ")
        start = time.time()
        for game_num in range(1, 1272):
            self._game_to_json(season_year, game_num, preseason)
            progress = game_num * 100 / 1271
            if int(progress) % 10 == 0 and abs(progress - int(progress) < 0.1):
                print("{}%".format(int(progress)), end=" ", flush=True)
        end = time.time()
        print("] done in {:.2f} s".format(end - start))


    def create_player_names_dict(self, filename='player_names_v2.json'):
        print("## DatasetManager V2 ##: Creating players 'IDs <-> NAMES' dictionary")
        player_names = {'id_to_name': {}, 'name_to_id': {}}
        for season in os.listdir("games"):
            start = time.time()
            print("season {}/{} ... ".format(season, int(season)+1), end="", flush=True)
            for game_filename in os.listdir("games/{}".format(season)):
                with open("games/{}/{}".format(season, game_filename), 'r') as f:
                    game_dict = json.load(f)
                for h_a in ['home', 'away']:
                    for player_id in game_dict['players'][h_a]:
                        if player_id not in player_names['id_to_name']:
                            player_json = requests.get(self._base_url + "people/{}".format(player_id)).json()
                            player_names['id_to_name'][player_id] = player_json['people'][0]['fullName']
            end = time.time()
            print("done in {:.2f} s".format(end - start))
        for player_id, player_name in player_names['id_to_name'].items():
            player_names['name_to_id'][player_name] = player_id
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(player_names, f, ensure_ascii=False, indent=4)


    def _get_player_name(self, player_id):
        if player_id is None:
            return None
        return self._player_names['id_to_name'][player_id]


    def _get_player_id(self, player_name):
        if player_name is None:
            return None
        return self._player_names['name_to_id'][player_name]


    def create_team_names_dict(self, filename='team_names_v2.json'):
        print("## DatasetManager V2 ##: Creating teams 'IDs <-> NAMES' dictionary")
        team_names = {'id_to_name': {}, 'name_to_id': {}}
        for season in os.listdir("games"):
            print("season {}/{} ... ".format(season, int(season)+1), end="", flush=True)
            for game_filename in os.listdir("games/{}".format(season)):
                with open("games/{}/{}".format(season, game_filename), 'r') as f:
                    game_dict = json.load(f)
                team_id = game_dict['home_team_id']
                team_name = game_dict['home_team']
                if team_id not in team_names['id_to_name']:
                    team_names['id_to_name'][team_id] = team_name
                    team_names['name_to_id'][team_name] = team_id
            print("done")
        with open(filename, 'w') as f:
            json.dump(team_names, f, indent=4)


    def _print_base_dataset_memory_usage(self):
        if self._base_dataset is None:
            print("Base dataset has not been created yet (is None).")
        else:
            print("Base dataset has: {:.2f} MB".format(self._base_dataset.memory_usage().sum() * 0.000001))

    
    def create_base_dataset(self, filename="base_dataset_v2.csv", icetime_last_n_games=5, names=True):
        print("## DatasetManager V2 ##: Creating Base dataset")
        # construct columns list
        columns = ['year', 'month', 'day', 'hour', 'home_team', 'away_team', 'result', 'home_coach', 'away_coach']
        for h_a in ['home', 'away']:
            columns.append('{}_G'.format(h_a))
            for i_dman in range(1, 9):
                columns.append('{}_D{}'.format(h_a, i_dman))
                columns.append('{}_D{}_time'.format(h_a, i_dman))
            for i_line in range(1, 5):   
                columns.append('{}_L{}-LW'.format(h_a, i_line))
                columns.append('{}_L{}-LW_time'.format(h_a, i_line))
                columns.append('{}_L{}-C'.format(h_a, i_line))
                columns.append('{}_L{}-C_time'.format(h_a, i_line))
                columns.append('{}_L{}-RW'.format(h_a, i_line))
                columns.append('{}_L{}-RW_time'.format(h_a, i_line))
            columns.append('{}_F13'.format(h_a))
            columns.append('{}_F13_time'.format(h_a))
        # create DataFrame
        self._base_dataset = pd.DataFrame(columns=columns)
        # insert values
        for season in sorted(os.listdir('games/v2')):
            print("--> season {}/{} ... ".format(season, int(season)+1), end='', flush=True)
            start = time.time()
            indices = []
            season_table = []
            icetimes = {}
            for game_filename in sorted(os.listdir('games/v2/{}'.format(season))): 
                with open('games/v2/{}/{}'.format(season, game_filename), 'r') as f:
                    game_json = json.load(f)
                game_array = []
                indices.append(int(game_json['game_id']))
                # non-player features (date, teams)
                for column in columns[:6]:
                    game_array.append(game_json[column])
                # result
                if game_json['reg_draw'] == 1:
                    game_array.append('draw')
                else:
                    game_array.append('home' if game_json['home_goals'] > game_json['away_goals'] else 'away')
                # coaches
                for column in ['home_coach', 'away_coach']:
                    game_array.append(game_json[column])
                # players
                for h_a in ['home', 'away']:
                    goalies = []
                    goalies_times = []
                    dmen = []
                    dmen_times = []
                    forwards = []
                    forwards_times = []
                    for player_id, player in game_json['players'][h_a].items():
                        if player['position'] == 'G':
                            goalies.append(player_id)
                            goalies_times.append(player['icetime'])
                        elif player['position'] == 'D':
                            dmen.append(player_id)
                            dmen_times.append(player['icetime'])
                        elif player['position'] in ['L', 'C', 'R']:
                            forwards.append(player_id)
                            forwards_times.append(player['icetime'])
                        else:
                            raise ValueError
                    goalies = np.array(goalies)[np.argsort(goalies_times)][::-1]
                    goalies_times = np.sort(goalies_times)[::-1]
                    forwards = np.array(forwards)[np.argsort(forwards_times)][::-1]
                    forwards_times = np.sort(forwards_times)[::-1]
                    dmen = np.array(dmen)[np.argsort(dmen_times)][::-1]
                    dmen_times = np.sort(dmen_times)[::-1]
                    # goalie
                    if names:
                        game_array.append(self._get_player_name(goalies[0]))
                    else:
                        game_array.append(goalies[0])
                    # Defensemen
                    for player_id, icetime in zip(dmen, dmen_times):
                        if names:
                            game_array.append(self._get_player_name(player_id))
                        else:
                            game_array.append(player_id)
                        if player_id not in icetimes:
                            game_array.append(icetime)
                            icetimes[player_id] = deque([], icetime_last_n_games)
                        else:
                            game_array.append(int(np.mean(icetimes[player_id])))
                        icetimes[player_id].append(icetime)
                    for i in range(8 - len(dmen)):
                        if names:
                            game_array.append('vacant')
                        else:
                            game_array.append(0)
                        game_array.append(0)
                    # Forwards
                    for i_line in range(4):
                        posts = {
                            'L': None,
                            'C': None,
                            'R': None
                        }
                        for player_id in forwards[3*i_line:3*i_line+3]:
                            position = game_json['players'][h_a][player_id]['position']
                            if posts[position] is None:
                                posts[position] = player_id
                        for post, player_id in posts.items():
                            if player_id is None:
                                for player_id in forwards[3*i_line:3*i_line+3]:
                                    if player_id not in posts.values():
                                        posts[post] = player_id
                                        break
                        for _, player_id in posts.items():
                            if player_id is None:
                                if names:
                                    game_array.append('vacant')
                                else:
                                    game_array.append(0)
                                game_array.append(0)
                            else:
                                if names:
                                    game_array.append(self._get_player_name(player_id))
                                else:
                                    game_array.append(player_id)
                                icetime = forwards_times[np.argwhere(forwards == player_id)][0][0]
                                if player_id not in icetimes:
                                    game_array.append(icetime)
                                    icetimes[player_id] = deque([], icetime_last_n_games)
                                else:
                                    game_array.append(int(np.mean(icetimes[player_id])))
                                icetimes[player_id].append(icetime)
                    if len(forwards) < 13:
                        if names:
                            game_array.append('vacant')
                        else:
                            game_array.append(0)
                        game_array.append(0)
                    else:
                        if names:
                            game_array.append(self._get_player_name(forwards[-1]))
                        else:
                            game_array.append(forwards[-1])
                        if forwards[-1] not in icetimes:
                            game_array.append(forwards_times[-1])
                            icetimes[forwards[-1]] = deque([], icetime_last_n_games)
                        else:
                            game_array.append(int(np.mean(icetimes[forwards[-1]])))
                        icetimes[forwards[-1]].append(forwards_times[-1])
                season_table.append(game_array)
            season_df = pd.DataFrame(season_table, columns=columns, index=indices)
            self._base_dataset = pd.concat([self._base_dataset, season_df])
            end = time.time()
            print("done in {:.1f} s".format(end - start))
        self._print_base_dataset_memory_usage()
        self._base_dataset.to_csv(filename)
    