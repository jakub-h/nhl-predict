import pandas as pd
import numpy as np
import json
import requests
import time
import sys
import os
from collections import deque
from sklearn.preprocessing import MinMaxScaler, LabelEncoder




class DatasetManager():
    def __init__(self,
                 base_url="https://statsapi.web.nhl.com/api/v1/",
                 base_dataset_fn="base_dataset_v3.csv"):
        
        print("## DatasetManager V3 ##: Initialization")
        self._base_url = base_url
        
        # Base dataset (full dataset, with team names, not standardized etc.)
        if base_dataset_fn is not None and base_dataset_fn in os.listdir("./"):
            self._base_dataset = pd.read_csv(base_dataset_fn, header=0, index_col=0)
            print("--> Base dataset loaded from: {}".format(base_dataset_fn))
        else:
            self._base_dataset = None
        
        self._columns_base = ['year', 'month', 'day', 'hour', 'home_team', 'away_team', 'result', 'home_coach', 'away_coach']
        self._stats = ['GF', 'GF-5on5', 'GF-PP', 'GF-SH', 'FO-T', 'FO-W', 'SF', 'SF-5on5',
                       'SF-PP', 'SF-SH', 'PP-O', 'PIM', 'BLK', 'TAW', 'GAW', 'HIT', 'PK-O', 'GA',
                       'GA-5on5', 'GA-PP', 'GA-SH', 'SA', 'SA-5on5', 'SA-PP', 'SA-SH']
    

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
            parsed_game[h_a] = {}
            parsed_game[h_a]['team_id'] = game_dict['gameData']['teams'][h_a]['id']
            parsed_game[h_a]['team'] = game_dict['gameData']['teams'][h_a]['triCode']
            parsed_game[h_a]['coach'] = game_dict['liveData']['boxscore']['teams'][h_a]['coaches'][0]['person']['fullName']
            team_stats = game_dict['liveData']['boxscore']['teams'][h_a]['teamStats']['teamSkaterStats']
            parsed_game[h_a]['GF'] = team_stats['goals']
            parsed_game[h_a]['GF-5on5'] = 0
            parsed_game[h_a]['GF-PP'] = int(team_stats['powerPlayGoals'])
            parsed_game[h_a]['GF-SH'] = 0
            parsed_game[h_a]['SF'] = team_stats['shots']
            parsed_game[h_a]['SF-5on5'] = 0
            parsed_game[h_a]['SF-PP'] = 0
            parsed_game[h_a]['SF-SH'] = 0
            parsed_game[h_a]['PIM'] = team_stats['pim']
            parsed_game[h_a]['PP-O'] = int(team_stats['powerPlayOpportunities'])
            parsed_game[h_a]['BLK'] = team_stats['blocked']
            parsed_game[h_a]['TAW'] = team_stats['takeaways']
            parsed_game[h_a]['GAW'] = team_stats['giveaways']
            parsed_game[h_a]['HIT'] = team_stats['hits']
            parsed_game[h_a]['FO-T'] = 0
            parsed_game[h_a]['FO-W'] = 0

            
        for h_a, other_team in [('home', 'away'), ('away', 'home')]:
            for player in game_dict['liveData']['boxscore']['teams'][h_a]['players'].values():
                if 'skaterStats' in player['stats']:
                    parsed_game[h_a]['FO-T'] += player['stats']['skaterStats']['faceoffTaken']
                    parsed_game[h_a]['FO-W'] += player['stats']['skaterStats']['faceOffWins']
                    parsed_game[h_a]['GF-SH'] += player['stats']['skaterStats']['shortHandedGoals']
                elif 'goalieStats' in player['stats']:
                    parsed_game[other_team]['SF-5on5'] += player['stats']['goalieStats']['evenShotsAgainst']
                    parsed_game[other_team]['SF-PP'] += player['stats']['goalieStats']['powerPlayShotsAgainst']
                    parsed_game[other_team]['SF-SH'] += player['stats']['goalieStats']['shortHandedShotsAgainst']
                    parsed_game[other_team]['GF-5on5'] += player['stats']['goalieStats']['evenShotsAgainst'] - player['stats']['goalieStats']['evenSaves']
        return parsed_game


    def _game_to_json(self, season_year, game_num, preseason=False):
        game = self._seek_game(season_year, game_num, preseason)
        game = self._parse_game(game)
        if game is None:
            return
        game_type = "02"
        if preseason:
            game_type = "01"
        with open('games/v3/{}/{}{:04d}.json'.format(season_year, game_type, game_num), 'w', encoding='utf-8') as f:
            json.dump(game, f, ensure_ascii=False, indent=4)


    def get_season(self, season_year, preseason=False):
        print("## DatasetManager V3 ##: Downloading season {}/{} (preseason = {})".format(season_year, season_year+1, preseason))
        print("[", end=" ")
        start = time.time()
        for game_num in range(1, 1272):
            self._game_to_json(season_year, game_num, preseason)
            progress = game_num * 100 / 1271
            if int(progress) % 10 == 0 and abs(progress - int(progress) < 0.1):
                print("{}%".format(int(progress)), end=" ", flush=True)
        end = time.time()
        print("] done in {:.2f} s".format(end - start))
    

    def _print_base_dataset_memory_usage(self):
        if self._base_dataset is None:
            print("Base dataset has not been created yet (is None).")
        else:
            print("Base dataset has: {:.2f} MB".format(self._base_dataset.memory_usage().sum() * 0.000001))


    def create_base_dataset(self, filename="base_dataset_v3.csv"):
        print("## DatasetManager V2 ##: Creating Base dataset")
        # construct columns list
        columns = self._columns_base.copy()
        
        for h_a in ['home', 'away']:
            for feature in self._stats:
                columns.append("{}_{}".format(h_a, feature))
        # create DataFrame
        self._base_dataset = pd.DataFrame(columns=columns)
        # insert values
        for season in sorted(os.listdir('games/v3')):
            print("--> season {}/{} ... ".format(season, int(season)+1), end='', flush=True)
            start = time.time()
            indices = []
            season_table = []
            for game_filename in sorted(os.listdir('games/v3/{}'.format(season))):
                with open('games/v3/{}/{}'.format(season, game_filename), 'r') as f:
                    game_json = json.load(f)
                game_array = []
                indices.append(int(game_json['game_id']))
                # date and time features
                for col in columns[:4]:
                    game_array.append(game_json[col])
                # team names
                game_array.append(game_json['home']['team'])
                game_array.append(game_json['away']['team'])
                # result (home, draw, away)
                if game_json['reg_draw'] == 1:
                    game_array.append('draw')
                else:
                    game_array.append('home' if game_json['home']['GF'] > game_json['away']['GF'] else 'away')
                game_array.append(game_json['home']['coach'])
                game_array.append(game_json['away']['coach'])
                # game statistics
                for h_a, other_team in [('home', 'away'), ('away', 'home')]:
                    for feature in self._stats[:16]:
                        game_array.append(game_json[h_a][feature])
                    # some co-statistics (PK-O from PP-O, GA and SA from GF and SA)
                    game_array.append(game_json[other_team]['PP-O'])
                    game_array.append(game_json[other_team]['GF'])
                    game_array.append(game_json[other_team]['GF-5on5'])
                    game_array.append(game_json[other_team]['GF-SH'])
                    game_array.append(game_json[other_team]['GF-PP'])
                    game_array.append(game_json[other_team]['SF'])
                    game_array.append(game_json[other_team]['SF-5on5'])
                    game_array.append(game_json[other_team]['SF-SH'])
                    game_array.append(game_json[other_team]['SF-PP'])
                    
                season_table.append(game_array)
            season_df = pd.DataFrame(season_table, columns=columns, index=indices)
            self._base_dataset = pd.concat([self._base_dataset, season_df])
            end = time.time()
            print("done in {:.1f} s".format(end - start))
        self._print_base_dataset_memory_usage()
        self._base_dataset.to_csv(filename)


    def _process_game_from_base_dataset(self, game, season_stats, form_stats, form_length):
        game_array = []
        for val in game[self._columns_base].tolist():
            game_array.append(val)
        for h_a in ['home', 'away']:
            team_name = game['{}_team'.format(h_a)]
            for stat in self._stats:
                if team_name not in season_stats[stat]:
                    season_stats[stat][team_name] = deque([], maxlen=82)
                    game_array.append(game["{}_{}".format(h_a, stat)])
                else:
                    game_array.append(np.mean(season_stats[stat][team_name]))
                season_stats[stat][team_name].append(game['{}_{}'.format(h_a, stat)])                
                if team_name not in form_stats[stat]:
                    form_stats[stat][team_name] = deque([], maxlen=form_length)
                    game_array.append(game['{}_{}'.format(h_a, stat)])
                else:
                    game_array.append(np.mean(form_stats[stat][team_name]))
                form_stats[stat][team_name].append(game['{}_{}'.format(h_a, stat)])
            if game['result'] == 'draw':
                form_points = 1
            elif game['result'] == 'home':
                form_points = 2 if h_a == 'home' else 0
            else:
                form_points = 2 if h_a == 'away' else 0
            if team_name not in form_stats['points']:
                form_stats['points'][team_name] = deque([], maxlen=form_length)
                game_array.append(0)
            else:
                game_array.append(np.mean(form_stats['points'][team_name]))
            form_stats['points'][team_name].append(form_points)
        return game_array
    

    def create_seasonal_split(self, test_season, first_games_to_train, num_of_train_seasons, form_length, dummy, home_rel, minmax):
        train_seasons = []
        for i in range(num_of_train_seasons):
            train_seasons.append(test_season-1-i)
        train_seasons = sorted(train_seasons, reverse=True)
        print(("## DatasetManager V3 ##: Create seasonal split (test seas.: {}; "
               "first games: {}; train seasons: {}; form: {}; dummy: {}; rel to home: {}; minmax: {})".format(
                    test_season, first_games_to_train, train_seasons, form_length, dummy, home_rel, minmax)))
        if self._base_dataset is None:
            print("You must load or create human readable dataset first!")
            return
        
        print("... ", end='', flush=True)
        start = time.time()
        # Prepare columns
        columns = self._columns_base.copy()
        for h_a in ['home', 'away']:
            for stat in self._stats:
                columns.append("{}_{}".format(h_a, stat))
                columns.append("{}_{}_form".format(h_a, stat))
            columns.append("{}_points_form".format(h_a))
        # TRAIN seasons
        train = []
        train_ind = []
        for i_season in train_seasons:
            season_df = self._base_dataset.loc[(self._base_dataset.index > int("{}020000".format(i_season))) &
                                               (self._base_dataset.index < int("{}020000".format(i_season+1)))]
            season_stats = {}
            form_stats = {}
            for stat in self._stats:
                season_stats[stat] = {}
                form_stats[stat] = {}
            form_stats['points'] = {}
            for game_id, game in season_df.iterrows():
                train.append(self._process_game_from_base_dataset(game, season_stats, form_stats, form_length))
                train_ind.append(game_id)
        # TEST season
        test = []
        test_ind = []
        test_season_df = self._base_dataset.loc[(self._base_dataset.index > int("{}020000".format(test_season))) &
                                                (self._base_dataset.index < int("{}020000".format(test_season+1)))]
        season_stats = {}
        form_stats = {}
        for stat in self._stats:
            season_stats[stat] = {}
            form_stats[stat] = {}
        form_stats['points'] = {}
        for game_id, game in test_season_df.iterrows():
            game_array = self._process_game_from_base_dataset(game, season_stats, form_stats, form_length)
            if game_id - (test_season * 1000000 + 20000) <= first_games_to_train:
                train.append(game_array)
                train_ind.append(game_id)
            else:
                test.append(game_array)
                test_ind.append(game_id)

        train = pd.DataFrame(data=train, index=train_ind, columns=columns)
        test = pd.DataFrame(data=test, index=test_ind, columns=columns)

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
            train = pd.get_dummies(train, columns=['home_team', 'away_team'])
            test = pd.get_dummies(test, columns=['home_team', 'away_team'])
            for missing_team in test.columns.difference(train.columns.values).values:
                train[missing_team] = np.zeros(train.shape[0], dtype=float)
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
        # Code stats as relative to the home team
        if home_rel:
            for stat in self._stats:
                for form in ['', '_form']:
                    train['{}{}'.format(stat, form)] = train['home_{}{}'.format(stat, form)] - train['away_{}{}'.format(stat, form)]
                    train.drop(columns=['home_{}{}'.format(stat, form), 'away_{}{}'.format(stat, form)], inplace=True, axis='columns')
                    test['{}{}'.format(stat, form)] = test['home_{}{}'.format(stat, form)] - test['away_{}{}'.format(stat, form)]
                    test.drop(columns=['home_{}{}'.format(stat, form), 'away_{}{}'.format(stat, form)], inplace=True, axis='columns')
            train['points_form'] = train['home_points_form'] - train['away_points_form']
            train.drop(columns=['home_points_form', 'away_points_form'], inplace=True, axis='columns')
            test['points_form'] = test['home_points_form'] - test['away_points_form']
            test.drop(columns=['home_points_form', 'away_points_form'], inplace=True, axis='columns')
        # Scale
        if minmax:
            scaler = MinMaxScaler()
            target = train['result']
            train = pd.DataFrame(scaler.fit_transform(train.drop(columns=['result'])),
                                index=train.index,
                                columns=np.delete(train.columns.values, np.argwhere(train.columns.values == 'result')))
            train = pd.concat([train, target], axis=1)
            target = test['result']
            test = pd.DataFrame(scaler.transform(test.drop(columns=['result'])),
                                index=test.index,
                                columns=np.delete(test.columns.values, np.argwhere(test.columns.values == 'result')))
            test = pd.concat([test, target], axis=1)
        # Save
        out_fn = "{}-{}_".format(str(test_season)[-2:], first_games_to_train)
        for season in train_seasons:
            out_fn += "{}".format(str(season)[-2:])
        out_fn += "_{:02d}".format(form_length)
        if dummy:
            out_fn += "_dm"
        if home_rel:
            out_fn += "_hrel"
        if minmax:
            out_fn += '_minmax'
        train.to_csv("seasonal_splits/v3/train_{}.csv".format(out_fn))
        test.to_csv("seasonal_splits/v3/test_{}.csv".format(out_fn))
        end = time.time()
        print("done in {:.2f} s".format(end - start))
    

    def get_seasonal_split(self, test_season, first_games_to_train, num_of_train_seasons, form_length, dummy, home_rel, minmax):
        fn = "{}-{}_".format(str(test_season)[-2:], first_games_to_train)
        train_seasons = []
        for i in range(num_of_train_seasons):
            train_seasons.append(test_season-1-i)
        train_seasons = sorted(train_seasons, reverse=True)
        for season in train_seasons:
            fn += str(season)[-2:]
        fn += "_{:02d}".format(form_length)
        if dummy:
            fn += "_dm"
        if home_rel:
            fn += "_hrel"
        if minmax:
            fn += '_minmax'
        try:
            train = pd.read_csv("seasonal_splits/v3/train_{}.csv".format(fn), index_col=0)
        except FileNotFoundError:
            print("## DatasetManager ##: file 'train_{}.csv' does not exist (Have you created folds already?)".format(fn))
            sys.exit()
        try:
            test = pd.read_csv("seasonal_splits/v3/test_{}.csv".format(fn), index_col=0)
        except FileNotFoundError:
            print("## DatasetManager ##: file 'test_{}.csv' does not exist (Have you created folds already?)".format(fn))
            sys.exit()
        return train.drop(columns=['result']), test.drop(columns=['result']), train['result'], test['result']



                        
