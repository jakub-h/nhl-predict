import pandas as pd
import json
import requests
import time
import os
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

BYTES_TO_MB_DIV = 0.000001

class DatasetManager():
    def __init__(self, base_url="https://statsapi.web.nhl.com/api/v1/",
                 player_names_fn="player_names.json",
                 team_names_fn="team_names.json",
                 games_hr_fn="human_readable.csv"):
        print("## DatasetManager ##: Initialization")
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

        # human readable dataset (full dataset, with team names, not standardized etc.)
        if games_hr_fn is not None and games_hr_fn in os.listdir("./"):
            self._games_hr = pd.read_csv(games_hr_fn, header=0, index_col=0)
            print("--> HR dataset loaded from: {}".format(games_hr_fn))
        else:
            self._games_hr = None
        
        self._team_encoder = None
        self._coach_encoder = None
        self._scaler = None

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
        print("## DatasetManager ##: Downloading season {}/{} (preseason = {})".format(season_year, season_year+1, preseason))
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
        print("## DatasetManager ##: Creating players 'IDs <-> NAMES' dictionary")
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
        print("## DatasetManager ##: Creating teams 'IDs <-> NAMES' dictionary")
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
        print("## DatasetManager ##: Creating HR dataset")
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

    
    def load_hr_from_csv(self, filename="human_readable.csv"):
        print("## DatasetManager ##: Loading HR dataset from csv ... ", end="", flush=True)
        self._games_hr = pd.read_csv(filename, header=0, index_col=0)
        print("done")   


    def create_k_folds(self, k, dummy_teams=False, dummy_coaches=False, scale=None):
        print("## DatasetManager ##: Creating {}-folds (dummy_teams={}, dummy_coaches={}, scale={})".format(k, dummy_teams, dummy_coaches, scale))
        if self._games_hr is None:
            print("You must load or create human readable dataset first!")
            return
        # shuffle indices and create folds: (train, test) pairs k-times
        indices = self._games_hr.index.to_list()
        np.random.shuffle(indices)
        folds_ind = np.array_split(indices, k)
        for fold_i in range(k):
            print("--> fold {} ... ".format(fold_i), end='', flush=True)
            start = time.time()
            folds_ind_copy = folds_ind.copy()
            test_ind = folds_ind_copy.pop(fold_i)
            train_ind = []
            for chunk in folds_ind_copy:
                for game_id in chunk:
                    train_ind.append(game_id)
            train_ind = np.array(train_ind)

            # TRAIN DATASET
            train = self._games_hr.copy().loc[train_ind]
            # GOALS -> result (1: home win, 0: regulation draw, -1:away win)
            train['result'] = 1 - train['reg_draw']
            goal_diff = train.loc[train['result'] == 1, 'home_goals'] - train.loc[train['result'] == 1, 'away_goals']
            train.loc[train['result'] == 1, 'result'] = np.sign(goal_diff)
            train.drop(columns=['home_goals', 'away_goals', 'reg_draw'], inplace=True)
            # TEAMS
            if dummy_teams:
                train = pd.get_dummies(train, prefix=['home', 'away'], columns=['home_team_name', 'away_team_name'])
            else:
                train.rename(columns={"home_team_name": "home_team", "away_team_name": "away_team"}, inplace=True)
                self._team_encoder = LabelEncoder()
                self._team_encoder.fit(sorted(train['home_team']))
                train['home_team'] = self._team_encoder.transform(train['home_team'])
                train['away_team'] = self._team_encoder.transform(train['away_team'])
            # COACHES
            # merge all coaches with 20 or less games to 'other' value
            coach_counts = train['home_coach'].value_counts() + train['away_coach'].value_counts()
            train['home_coach'] = train['home_coach'].apply(lambda coach_name: coach_name if coach_counts[coach_name] > 20 else "other")
            train['away_coach'] = train['away_coach'].apply(lambda coach_name: coach_name if coach_counts[coach_name] > 20 else "other")
            # encoding
            self._coach_encoder = LabelEncoder()
            self._coach_encoder.fit(np.concatenate([train['home_coach'], train['away_coach']]))
            train['home_coach'] = self._coach_encoder.transform(train['home_coach'])
            train['away_coach'] = self._coach_encoder.transform(train['away_coach'])
            if dummy_coaches:
                train = pd.get_dummies(train, columns=['home_coach', 'away_coach'])
            # PLAYERS
            # merge all players with 100 or less minutes to 'other' value
            other_players = []
            for player_id in self._player_names['id_to_name'].keys():
                if (train["{}_home".format(player_id)].sum() + train['{}_away'.format(player_id)].sum()) <= 6000:
                    other_players.append(player_id)
            train['other_home'] = np.zeros(train.shape[0], dtype=int)
            train['other_away'] = np.zeros(train.shape[0], dtype=int)
            for player_id in other_players:
                train['other_home'] += train['{}_home'.format(player_id)]
                train['other_away'] += train['{}_away'.format(player_id)]
                train.drop(columns=['{}_home'.format(player_id), '{}_away'.format(player_id)], inplace=True)
            # SCALING
            if scale == 'std':
                self._scaler = StandardScaler()
            if scale == 'minmax':
                self._scaler = MinMaxScaler()
            if scale is not None:
                target = train['result']
                train = pd.DataFrame(self._scaler.fit_transform(train.drop(columns=['result'])),
                                                                index=train.index,
                                                                columns=np.delete(train.columns.values,
                                                                                  np.argwhere(train.columns.values == 'result')))
                train = pd.concat([train, target], axis=1)
            # SAVE
            out_fn = str(fold_i)
            if dummy_teams:
                out_fn += "_dt"
            if dummy_coaches:
                out_fn += "_dc"
            if scale is None:
                out_fn += "_noscale"
            else:
                out_fn += "_{}".format(scale)
            train.to_csv("folds/train_{}.csv".format(out_fn))

            # TEST DATASET
            test = self._games_hr.copy().loc[test_ind]
            # GOALS -> result (1: home win, 0: regulation draw, -1: away win)
            test['result'] = 1 - test['reg_draw']
            goal_diff = test.loc[test['result'] == 1, 'home_goals'] - test.loc[test['result'] == 1, 'away_goals']
            test.loc[test['result'] == 1, 'result'] = np.sign(goal_diff)
            test.drop(columns=['home_goals', 'away_goals', 'reg_draw'], inplace=True)
            # TEAMS
            if dummy_teams:
                test = pd.get_dummies(test, prefix=['home', 'away'], columns=['home_team_name', 'away_team_name'])
            else:
                test.rename(columns={"home_team_name": "home_team", "away_team_name": "away_team"}, inplace=True)
                test['home_team'] = self._team_encoder.transform(test['home_team'])
                test['away_team'] = self._team_encoder.transform(test['away_team'])     
            # COACHES
            # mark all coaches not presented in train's schema as 'other'
            test['home_coach'] = test['home_coach'].apply(lambda coach_name: coach_name if coach_name in self._coach_encoder.classes_ else "other")
            test['away_coach'] = test['away_coach'].apply(lambda coach_name: coach_name if coach_name in self._coach_encoder.classes_ else "other")
            # encoding
            test['home_coach'] = self._coach_encoder.transform(test['home_coach'])
            test['away_coach'] = self._coach_encoder.transform(test['away_coach'])
            # dummy variables and missing coaches (presented in train, missing in test)
            if dummy_coaches:
                test = pd.get_dummies(test, columns=['home_coach', 'away_coach'])
                for missing_coach in train.columns.difference(test.columns.values).values:
                    test[missing_coach] = np.zeros(test.shape[0], dtype=int)
            # PLAYERS
            # mark all players not presented in train's schema as 'other'
            test['other_home'] = np.zeros(test.shape[0], dtype=int)
            test['other_away'] = np.zeros(test.shape[0], dtype=int)
            for player_id in other_players:
                test['other_home'] += test['{}_home'.format(player_id)]
                test['other_away'] += test['{}_away'.format(player_id)]
                test.drop(columns=['{}_home'.format(player_id), '{}_away'.format(player_id)], inplace=True)
            # SCALING
            if scale is not None:
                target = test['result']
                test = pd.DataFrame(self._scaler.transform(test.drop(columns=['result'])),
                                                           index=test.index,
                                                           columns=np.delete(test.columns.values,
                                                                             np.argwhere(test.columns.values == 'result')))
                test = pd.concat([test, target], axis=1)
            # SAVE
            test.to_csv("folds/test_{}.csv".format(out_fn))
            end = time.time()
            print("done in {:.2f} s".format(end - start))


    def get_fold(self, i, dummy_teams=False, dummy_coaches=False, scale=None):
        fn = "{}".format(i)
        if dummy_teams:
            fn += "_dt"
        if dummy_coaches:
            fn += "_dc"
        if scale is None:
            fn += "_noscale"
        else:
            fn += "_{}".format(scale)
        try:
            train = pd.read_csv("folds/train_{}.csv".format(fn), index_col=0)
        except FileNotFoundError:
            print("## DatasetManager ##: file 'train_{}.csv' does not exist (Have you created folds already?)".format(fn))
            sys.exit()
        try:
            test = pd.read_csv("folds/test_{}.csv".format(fn), index_col=0)
        except FileNotFoundError:
            print("## DatasetManager ##: file 'test_{}.csv' does not exist (Have you created folds already?)".format(fn))
            sys.exit()
        return train.drop(columns=['result']), test.drop(columns=['result']), train['result'], test['result']
        

       
    
    
        