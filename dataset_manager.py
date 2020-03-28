import pandas as pd
import json
import requests
import time
import os
import numpy as np

BYTES_TO_MB_DIV = 0.000001

class DatasetManager():
    def __init__(self, base_url="https://statsapi.web.nhl.com/api/v1/"):
        self._base_url = base_url
        self._games_hr = None    # human readable
        self._games_ml = None    # for ML purposes (one-hot encoding, sparse matrix, standardized, ...)


    def get_hr_table(self):
        return self._games_hr 


    def _seek_game(self, season_year, game_num, preseason=False):
        game_type = "02"
        if preseason:
            game_type = "01"
        game_url = self._base_url + "game/{}{}{:04d}/feed/live".format(season_year, game_type, int(game_num))
        response = requests.get(game_url)
        return response.json()


    def _parse_game(self, game_dict):
        parsed_game = {}
        if 'gamePk' not in game_dict.keys():
            return None
        if game_dict['gameData']['status']['abstractgameState'] != "Final":
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
            parsed_game['{}_team_name'.format(h_a)] = game_dict['gameData']['teams'][h_a]['triCode']
            parsed_game['{}_goals'.format(h_a)] = game_dict['liveData']['linescore']['teams'][h_a]['goals']
            parsed_game['{}_coach'.format(h_a)] = game_dict['liveData']['boxscore']['teams'][h_a]['coaches'][0]['person']['fullName']
        parsed_game['players'] = {'home': {}, 'away': {}}
        for h_a in ['home', 'away']:
            for player_type in ['goalie', 'skater']:
                for player_id in game_dict['liveData']['boxscore']['teams'][h_a]['{}s'.format(player_type)]:
                    if player_id not in game_dict['liveData']['boxscore']['teams'][h_a]['scratches']:
                        raw_icetime = game_dict['liveData']['boxscore']['teams'][h_a]['players']['ID{}'.format(player_id)]['stats']['{}Stats'.format(player_type)]['timeOnIce']
                        minutes, seconds = raw_icetime.split(":")
                        parsed_game['players'][h_a][player_id] = 60*int(minutes) + int(seconds)
        return parsed_game
    

    def _game_to_json(self, season_year, game_num, preseason=False):
        game = self._seek_game(season_year, game_num, preseason)
        game = self._parse_game(game)
        if game is None:
            return
        game_type = "02"
        if preseason:
            game_type = "01"
        with open('games/{}/{}{:04d}.json'.format(season_year, game_type, game_num), 'w', encoding='utf-8') as f:
            json.dump(game, f, ensure_ascii=False, indent=4)


    def get_season(self, season_year, preseason=False):
        print("### DOWNLOADING SEASON {}/{} (preseason = {}) in progress...".format(season_year, season_year+1, preseason))
        print("[", end=" ")
        start = time.time()
        for game_num in range(1, 1272):
            self._game_to_json(season_year, game_num, preseason)
            progress = game_num * 100 / 1271
            if int(progress) % 10 == 0 and abs(progress - int(progress) < 0.1):
                print("{}%".format(int(progress)), end=" ", flush=True)
        end = time.time()
        print("] done in {:.2f} s".format(end - start))


    def create_player_names_dict(self):
        print("### CREATING PLAYERS 'IDs <-> NAMES' DICTIONARY in progress...")
        player_names = {'id_to_name': {}, 'name_to_id': {}}
        for season in os.listdir("games"):
            start = time.time()
            print("season {}/{} ... ".format(season, int(season)+1), end="", flush=True)
            for game_filename in os.listdir("games/{}".format(season)):
                with open("games/{}/{}".format(season, game_filename), 'r') as f:
                    game_dict = json.load(f)
                for h_a in ['home', 'away']:
                    for player_id in game_dict['players'][h_a].keys():
                        if player_id not in player_names['id_to_name'].keys():
                            player_json = requests.get(self._base_url + "people/{}".format(player_id)).json()
                            player_names['id_to_name'][player_id] = player_json['people'][0]['fullName']
            end = time.time()
            print("done in {:.2f} s".format(end - start))
        for player_id, player_name in player_names['id_to_name'].items():
            player_names['name_to_id'][player_name] = player_id
        with open("player_names.json", 'w', encoding="utf-8") as f:
            json.dump(player_names, f, ensure_ascii=False, indent=4)


    def create_team_names_dict(self):
        print("### CREATING TEAMS 'IDs <-> NAMES' DICTIONARY in progress...")
        team_names = {'id_to_name': {}, 'name_to_id': {}}
        for season in os.listdir("games"):
            print("season {}/{} ... ".format(season, int(season)+1), end="", flush=True)
            for game_filename in os.listdir("games/{}".format(season)):
                with open("games/{}/{}".format(season, game_filename), 'r') as f:
                    game_dict = json.load(f)
                team_id = game_dict['home_team_id']
                team_name = game_dict['home_team_name']
                if team_id not in team_names['id_to_name'].keys():
                    team_names['id_to_name'][team_id] = team_name
                    team_names['name_to_id'][team_name] = team_id
            print("done")
        with open("team_names.json", 'w') as f:
            json.dump(team_names, f, indent=4)


    def _print_hr_memory_usage(self):
        if self._games_hr is None:
            print("Human Readable dataset has not been created yet (is None).")
        else:
            print("Human Readable dataset has: {:.2f} MB".format(self._games_hr.memory_usage().sum() * BYTES_TO_MB_DIV))


    def create_hr_table(self):
        print("### CREATING HUMAN READABLE TABLE in progress...")
        print("schema ... ", end="", flush=True)
        start = time.time()
        # construct columns list
        columns = ['year', 'month', 'day', 'hour', 'home_team_name', 'away_team_name',
                   'home_goals', 'away_goals', 'reg_draw', 'home_coach', 'away_coach']
        with open("player_names.json", 'r') as f:
            names_dict = json.load(f)
        for player_id in names_dict['id_to_name'].keys():
            for h_a in ['home', 'away']:
                columns.append("{}_{}".format(player_id, h_a))
        # create DataFrame
        self._games_hr = pd.DataFrame(columns=columns)
        end = time.time()
        print("done in {:.1f} s".format(end - start))
        # insert values
        for season in sorted(os.listdir('games')):
            print("--> season {}/{} ... ".format(season, int(season)+1), end='', flush=True)
            start = time.time()
            indices = []
            season_table = []
            for game_filename in sorted(os.listdir('games/{}'.format(season))):
                with open('games/{}/{}'.format(season, game_filename), 'r') as f:
                    game_json = json.load(f)
                game_array = []
                indices.append(int(game_json['game_id']))
                # non-player features (date, teams, coaches, ...)
                for column in columns[:11]:
                    game_array.append(game_json[column])
                # players (every player in home and away version)
                for column in columns[11:]:
                    player_id, h_a = column.split('_')
                    if player_id in game_json['players'][h_a].keys():
                        game_array.append(game_json['players'][h_a][player_id])
                    else:
                        game_array.append(0)
                season_table.append(game_array)
            season_df = pd.DataFrame(season_table, columns=columns, index=indices)
            self._games_hr = pd.concat([self._games_hr, season_df])
            end = time.time()
            print("done in {:.1f} s".format(end - start))
        self._print_hr_memory_usage()

    
    def hr_to_csv(self, filename="human_readable.csv"):
        if self._games_hr is None:
            print("Human Readable dataset has not been created yet (is None).")
        else:
            self._games_hr.to_csv(filename)

    
    def hr_from_csv(self, filename="human_readable.csv"):
        self._games_hr = pd.read_csv(filename, header=0, index_col=0)
    
       
    
    
        