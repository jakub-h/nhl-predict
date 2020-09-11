import pandas as pd
import json
import requests
import time
import os
import sys
import numpy as np
from collections import deque
from sklearn.preprocessing import LabelEncoder, StandardScaler



class DatasetManager():
    def __init__(self, base_url="https://statsapi.web.nhl.com/api/v1/",
                 player_names_fn="player_names.json",
                 base_dataset_fn="base_dataset_v1.csv"):
        print("## DatasetManager V1 ##: Initialization")
        # Base URL for NHL API
        self._base_url = base_url

        # Players dictionary (IDs to names and vice versa)
        if player_names_fn is not None and player_names_fn in os.listdir("./"):
            with open(player_names_fn, 'r') as f:
                self._player_names = json.load(f)
            print("--> players 'ID <-> name' dictionary loaded from: {}".format(player_names_fn))
        else:
            self._player_names = None

        # Base dataset (full dataset, with team names, not standardized etc.)
        if base_dataset_fn is not None and base_dataset_fn in os.listdir("./"):
            self._base_dataset = pd.read_csv(base_dataset_fn, header=0, index_col=0)
            print("--> Base dataset loaded from: {}".format(base_dataset_fn))
        else:
            self._base_dataset = None
        
        # Base columns
        self._base_columns = ['year', 'month', 'day', 'hour', 'home_team', 'away_team', 'result', 'home_coach', 'away_coach']


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
        print("## DatasetManager V1 ##: Downloading season {}/{} (preseason = {})".format(season_year, season_year+1, preseason))
        print("[", end=" ")
        start = time.time()
        for game_num in range(1, 1272):
            self._game_to_json(season_year, game_num, preseason)
            progress = game_num * 100 / 1271
            if int(progress) % 10 == 0 and abs(progress - int(progress) < 0.1):
                print("{}%".format(int(progress)), end=" ", flush=True)
        end = time.time()
        print("] done in {:.2f} s".format(end - start))


    def create_player_names_dict(self, filename='player_names.json'):
        print("## DatasetManager V1 ##: Creating players 'IDs <-> NAMES' dictionary")
        player_names = {'id_to_name': {}, 'name_to_id': {}}
        for season in os.listdir("games/v1"):
            start = time.time()
            print("season {}/{} ... ".format(season, int(season)+1), end="", flush=True)
            for game_filename in os.listdir("games/v1/{}".format(season)):
                with open("games/v1/{}/{}".format(season, game_filename), 'r') as f:
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


    def _print_base_dataset_memory_usage(self):
        if self._base_dataset is None:
            print("Base dataset has not been created yet (is None).")
        else:
            print("Base dataset has: {:.2f} MB".format(self._base_dataset.memory_usage().sum() * 0.000001))


    def create_base_dataset(self, filename="base_dataset_v1.csv"):
        print("## DatasetManager V1 ##: Creating Base dataset")
        print("schema ... ", end="", flush=True)
        start = time.time()
        # construct columns list
        columns = self._base_columns.copy()
        for h_a in ['home', 'away']:
            for player_id in self._player_names['id_to_name']:
                columns.append("{}_{}".format(h_a, player_id))
        # create DataFrame
        self._base_dataset = pd.DataFrame(columns=columns)
        end = time.time()
        print("done in {:.1f} s".format(end - start))
        # insert values
        for season in sorted(os.listdir('games/v1')):
            print("--> season {}/{} ... ".format(season, int(season)+1), end='', flush=True)
            start = time.time()
            indices = []
            season_table = []
            for game_fn in sorted(os.listdir('games/v1/{}'.format(season))):
                with open('games/v1/{}/{}'.format(season, game_fn), 'r') as f:
                    game_json = json.load(f)
                game_array = []
                indices.append(int(game_json['game_id']))
                # date and teams
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
                # players (every player in home and away version)
                for column in columns[9:]:
                    h_a, player_id = column.split('_')
                    if player_id in game_json['players'][h_a].keys():
                        game_array.append(game_json['players'][h_a][player_id])
                    else:
                        game_array.append(0)
                season_table.append(game_array)
            season_df = pd.DataFrame(season_table, columns=columns, index=indices)
            self._base_dataset = pd.concat([self._base_dataset, season_df])
            end = time.time()
            print("done in {:.1f} s".format(end - start))
        self._print_base_dataset_memory_usage()
        self._base_dataset.to_csv(filename)  


    def _process_game_from_base_dataset(self, game, season_icetime, last_n, last_n_icetime):
        game_array = []
        for column, value in game.iteritems():
            if column in self._base_columns:
                game_array.append(value)
            else:
                player_id = column.split("_")[1]
                if value > 0:
                    if player_id not in season_icetime:
                        season_icetime[player_id] = deque([], maxlen=82)
                        last_n_icetime[player_id] = deque([], maxlen=last_n)
                        game_array.append(value)    # first to seasonal average
                        game_array.append(value)    # second to last N average
                    else:
                        game_array.append(np.mean(season_icetime[player_id]))
                        game_array.append(np.mean(last_n_icetime[player_id]))
                    season_icetime[player_id].append(value)
                    last_n_icetime[player_id].append(value)
                else:
                    game_array.append(0)    # first to seasonal average
                    game_array.append(0)    # second to last N average
        return game_array



    def create_seasonal_split(self, test_season, first_games_to_train, num_of_train_seasons, last_n, dummy):
        train_seasons = []
        for i in range(num_of_train_seasons):
            train_seasons.append(test_season-1-i)
        train_seasons = sorted(train_seasons, reverse=True)
        print(("## DatasetManager V1 ##: Create seasonal split (test seas.: {}; "
               "first games: {}; train seasons: {}; last_n: {}; dummy: {})".format(
                    test_season, first_games_to_train, train_seasons, last_n, dummy)))
        if self._base_dataset is None:
            print("You must load or create human readable dataset first!")
            return

        print("... ", end='', flush=True)
        start = time.time()        

        columns = self._base_columns.copy()
        for h_a in ['home', 'away']:
            for player_id in self._player_names['id_to_name']:
                for seas_last in ['', '_last-N']:
                    columns.append("{}_{}{}".format(h_a, player_id, seas_last))
        # TRAIN seasons
        train = []
        train_ind = []
        for i_season in train_seasons:
            season_df = self._base_dataset.loc[(self._base_dataset.index > int("{}020000".format(i_season))) &
                                               (self._base_dataset.index < int("{}020000".format(i_season+1)))]
            season_icetime = {}
            last_n_icetime = {}
            for game_id, game in season_df.iterrows():
                train.append(self._process_game_from_base_dataset(game, season_icetime, last_n, last_n_icetime))
                train_ind.append(game_id)

        # TEST season
        test = []
        test_ind = []
        test_season_df = self._base_dataset.loc[(self._base_dataset.index > int("{}020000".format(test_season))) &
                                                (self._base_dataset.index < int("{}020000".format(test_season+1)))]
        season_icetime = {}
        last_n_icetime = {}
        for game_id, game in test_season_df.iterrows():
            game_array = self._process_game_from_base_dataset(game, season_icetime, last_n, last_n_icetime)
            if game_id - (test_season * 1000000 + 20000) <= first_games_to_train:
                train.append(game_array)
                train_ind.append(game_id)
            else:
                test.append(game_array)
                test_ind.append(game_id)

        train = pd.DataFrame(data=train, index=train_ind, columns=columns)
        test = pd.DataFrame(data=test, index=test_ind, columns=columns)
        
        # merge all players with 100 or less minutes in train set to 'other' column
        other_players = []
        for player_id in self._player_names['id_to_name'].keys():
            if (train["home_{}".format(player_id)].sum() + train['away_{}'.format(player_id)].sum()) <= 6000:
                other_players.append(player_id)
        train['home_other_players'] = np.zeros(train.shape[0], dtype=float)
        train['away_other_players'] = np.zeros(train.shape[0], dtype=float)
        for player_id in other_players:
            train['home_other_players'] += train['home_{}'.format(player_id)]
            train['away_other_players'] += train['away_{}'.format(player_id)]
            train.drop(columns=['home_{}'.format(player_id), 'away_{}'.format(player_id),
                                'home_{}_last-N'.format(player_id), 'away_{}_last-N'.format(player_id)], inplace=True)        

        # mark all players in test set not presented in train's schema as 'other'
        test['home_other_players'] = np.zeros(test.shape[0], dtype=float)
        test['away_other_players'] = np.zeros(test.shape[0], dtype=float)
        for player_id in other_players:
            test['home_other_players'] += test['home_{}'.format(player_id)]
            test['away_other_players'] += test['away_{}'.format(player_id)]
            test.drop(columns=['home_{}'.format(player_id), 'away_{}'.format(player_id),
                               'home_{}_last-N'.format(player_id), 'away_{}_last-N'.format(player_id)], inplace=True)    


        # Mark coaches with 20 or less games as 'other' in train set
        coach_counts = train['home_coach'].value_counts() + train['away_coach'].value_counts()
        train['home_coach'] = train['home_coach'].apply(lambda coach_name: coach_name if coach_counts[coach_name] > 20 else "other")
        train['away_coach'] = train['away_coach'].apply(lambda coach_name: coach_name if coach_counts[coach_name] > 20 else "other")
        coach_encoder = LabelEncoder().fit(np.concatenate([train['home_coach'], train['away_coach'], ['other']]))
        # Mark coaches not presented in train dataset schema as 'other'
        test['home_coach'] = test['home_coach'].apply(lambda coach_name: coach_name if coach_name in coach_encoder.classes_ else "other")
        test['away_coach'] = test['away_coach'].apply(lambda coach_name: coach_name if coach_name in coach_encoder.classes_ else "other")
        # Dummy variables and labeling
        if dummy:
            # Teams
            train = pd.get_dummies(train, columns=['home_team', 'away_team'])
            test = pd.get_dummies(test, columns=['home_team', 'away_team'])
            for missing_team in test.columns.difference(train.columns.values).values:
                train[missing_team] = np.zeros(train.shape[0], dtype=float)
            # Coaches
            train = pd.get_dummies(train, columns=['home_coach', 'away_coach'])
            test = pd.get_dummies(test, columns=['home_coach', 'away_coach'])
            for missing_coach in train.columns.difference(test.columns.values).values:
                test[missing_coach] = np.zeros(test.shape[0], dtype=float)
            for missing_coach in test.columns.difference(train.columns.values).values:
                train[missing_coach] = np.zeros(train.shape[0], dtype=float)
        else:
            # Teams
            team_encoder = LabelEncoder().fit(np.concatenate([train['home_team'], test['home_team']]))
            train['home_team'] = team_encoder.transform(train['home_team'])
            train['away_team'] = team_encoder.transform(train['away_team'])
            test['home_team'] = team_encoder.transform(test['home_team'])
            test['away_team'] = team_encoder.transform(test['away_team'])
            # Coaches
            train['home_coach'] = coach_encoder.transform(train['home_coach'])
            train['away_coach'] = coach_encoder.transform(train['away_coach'])
            test['home_coach'] = coach_encoder.transform(test['home_coach'])
            test['away_coach'] = coach_encoder.transform(test['away_coach'])
        
        # Save
        out_fn = "{}-{}_".format(str(test_season)[-2:], first_games_to_train)
        for season in train_seasons:
            out_fn += "{}".format(str(season)[-2:])
        out_fn += "_l{}".format(last_n)
        if dummy:
            out_fn += "_dm"
        train.to_csv("seasonal_splits/v1/train_{}.csv".format(out_fn))
        test.to_csv("seasonal_splits/v1/test_{}.csv".format(out_fn))
        end = time.time()
        print("done in {:.2f} s".format(end - start))
        

    def get_seasonal_split(self, test_season, first_games_to_train, num_of_train_seasons, last_n, dummy):
        fn = "{}-{}_".format(str(test_season)[-2:], first_games_to_train)
        train_seasons = []
        for i in range(num_of_train_seasons):
            train_seasons.append(test_season-1-i)
        train_seasons = sorted(train_seasons, reverse=True)
        for season in train_seasons:
            fn += "{}".format(str(season)[-2:])
        fn += "_l{}".format(last_n)
        if dummy:
            fn += "_dm"
        try:
            train = pd.read_csv("seasonal_splits/v1/train_{}.csv".format(fn), index_col=0)
        except FileNotFoundError:
            print("## DatasetManager ##: file 'train_{}.csv' does not exist (Have you created folds already?)".format(fn))
            sys.exit()
        try:
            test = pd.read_csv("seasonal_splits/v1/test_{}.csv".format(fn), index_col=0)
        except FileNotFoundError:
            print("## DatasetManager ##: file 'test_{}.csv' does not exist (Have you created folds already?)".format(fn))
            sys.exit()
        return train.drop(columns=['result']), test.drop(columns=['result']), train['result'], test['result']

       
    
    
        