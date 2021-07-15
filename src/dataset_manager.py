import pandas as pd
from pathlib import Path
import pickle
from src.xg_model_base import XGModel
import matplotlib.pyplot as plt


class DatasetManager:
    """
    In-game stats (post game)
    Season stats (pre game)
    """
    def __init__(self, data_path):
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(f"'data_path' must be a string or instance of Path class. Not {type(data_path)}.")

    def calculate_post_game_stats(self, season, save_to_csv=False):
        """
        - away_ / home_
        - _EV, _PP, _SH
        - G, SOG, SMISS, HIT, TAKE, GIVE, FOW, PPO, PIM, BLK, CORSI
        - xG
        :param season:
        :return:
        """
        # Load games raw play-by-play
        with open(self._data_path / "games_raw" / f"{season}-{season+1}.pickle", 'rb') as f:
            games_raw = pickle.load(f)

        # Load xG play-by-play
        xg_pbp = pd.read_csv(self._data_path / "xg_pbp" / f"{season}-{season+1}.csv", index_col=0)

        # Load xG model
        xg_model = XGModel(self._data_path, model="../models/xg_extra_trees_n10")

        # Calculate xG predictions for all events (all games in the season)
        xg_pred = xg_model.predict(xg_pbp)
        xg_pbp['xg_pred'] = xg_pred

        # Define stats scheme
        stat_names = ["G", "SOG", "SMISS", "HIT", "TAKE", "GIVE", "FOW", "BLK", "CORSI", "xG"]
        schema = {
            "away_team_id": None,
            "home_team_id": None
        }
        for team in ['away', 'home']:
            for situation in ['ALL', 'EV', 'PP', 'SH']:
                for stat in stat_names:
                    schema[f"{team}_{stat}_{situation}"] = 0
            schema[f"{team}_PPO"] = 0

        games = []
        for raw_game in games_raw:
            game_stats = schema.copy()
            # Set team ids
            for team in ['away', 'home']:
                game_stats[f'{team}_team_id'] = raw_game['teams'][team]['id']
            # Calculate the stats
            for play in raw_game['plays']:
                if play['period'] <= 3:
                    self._process_play_post_game(play, raw_game['teams'], game_stats)

            # Calculate CORSI
            for team in ['away', 'home']:
                for situation in ['ALL', 'EV', 'PP', 'SH']:
                    game_stats[f"{team}_CORSI_{situation}"] = game_stats[f"{team}_G_{situation}"] +\
                                                              game_stats[f"{team}_SOG_{situation}"] +\
                                                              game_stats[f"{team}_SMISS_{situation}"]

                    # Add blocked shots (BLK is number of blocks of opposing team)
                    if team == "away":
                        game_stats[f"away_CORSI_{situation}"] += game_stats[f"home_BLK_{situation}"]
                    else:
                        game_stats[f"home_CORSI_{situation}"] += game_stats[f"away_BLK_{situation}"]

            for team in ['away', 'home']:
                for situation in ['ALL', 'EV', 'PP', 'SH']:
                    # Filter subset of events
                    home_flag = 1 if team == "home" else 0
                    subset = xg_pbp[(xg_pbp['game_id'] == raw_game['id']) & (xg_pbp['is_home'] == home_flag)]
                    if situation == "EV":
                        subset = subset[subset['strength_active'] == subset['strength_opp']]
                    elif situation == "PP":
                        subset = subset[subset['strength_active'] > subset['strength_opp']]
                    elif situation == "SH":
                        subset = subset[subset['strength_active'] < subset['strength_opp']]

                    # Calculate sum of xGs for events
                    xg_sum = 0 if subset.empty else subset['xg_pred'].sum()
                    game_stats[f'{team}_xG_{situation}'] = xg_sum

            games.append(game_stats)
        df = pd.DataFrame(games)
        df.index += 1
        if save_to_csv:
            df.to_csv(self._data_path / "games_stats" / "post_game" / f"{season}-{season+1}.csv")
        else:
            return df

    @staticmethod
    def _process_play_post_game(play, teams, game_stats):
        """
        Calculate post-game statistics from plays
        :param play:
        :param game_stats:
        :return:
        """
        # Determine active team
        if play['team']['id'] == teams['away']['id']:
            team = "away"
        else:
            team = "home"

        # Determine type of the stat
        if play['type'] == "GOAL":
            stat = "G"
        elif play['type'] == "SHOT":
            stat = "SOG"
        elif play['type'] == "MISSED_SHOT":
            stat = "SMISS"
        elif play['type'] == "HIT":
            stat = "HIT"
        elif play['type'] == "TAKEAWAY":
            stat = "TAKE"
        elif play['type'] == "GIVEAWAY":
            stat = "GIVE"
        elif play['type'] == "BLOCKED_SHOT":
            stat = "BLK"
        elif play['type'] == "FACEOFF":
            stat = "FOW"
        elif play['type'] == "PENALTY":
            stat = "PPO"
            team = "away" if team == "home" else "home"     # switch team (penalty on vs. powerplay opportunity)
        else:
            raise ValueError(f"W: Unknown type of play: {play['type']}")

        # Determine strength situation (even, pp, sh)
        if play['strength_active'] == play['strength_opp']:
            situation = "EV"
        elif play['strength_active'] > play['strength_opp']:
            situation = "PP"
        else:
            situation = "SH"

        # Increase the stat
        if stat != "PPO":
            game_stats[f"{team}_{stat}_{situation}"] += 1
            game_stats[f"{team}_{stat}_ALL"] += 1
        else:
            game_stats[f"{team}_PPO"] += 1

    def create_pre_game_dataset(self, season, last_n_games):
        """
        Creates a dataset with pre-game stats (from post-game stats).

        With average from whole season and average from last n games.
        :param season:
        :param last_n_games:
        :return:
        """
        pass
