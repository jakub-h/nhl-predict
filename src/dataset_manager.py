import pickle
import tqdm
from pathlib import Path

import pandas as pd

from src.xg_model_base import XGModel
import src.stats_utils as su

SITUATIONS = ["ALL", "EV", "PP", "SH"]


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

    def calculate_post_game_stats(self, season, save_to_csv=False): # noqa C901
        """
        Create a dataset with post-game stats (actual stats from each game).

        - [away_ | home_] ... prefix
        - [_ALL | _EV | _PP | _SH] ... suffix
        - G, SOG, SMISS, HIT, TAKE, GIVE, FOW, PPO, PIM, BLK, CORSI, xG ... stats

        :param season:
        :return:
        """
        # Load games raw play-by-play
        with open(self._data_path / "games_raw" / f"{season}-{season+1}.pickle", 'rb') as f:
            games_raw = pickle.load(f)

        # Load xG play-by-play
        xg_pbp = pd.read_csv(self._data_path / "xg_pbp" / f"{season}-{season+1}.csv", index_col=0)

        # Load xG model
        xg_model = XGModel(self._data_path, model=self._data_path.parent / "models/xg_extra_trees_n10")

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
            for situation in SITUATIONS:
                for stat in stat_names:
                    schema[f"{team}_{stat}_{situation}"] = 0
            schema[f"{team}_PPO"] = 0

        # Fill in stats
        games = []
        for raw_game in games_raw:
            game_stats = schema.copy()
            # Set team ids
            for team in ['away', 'home']:
                game_stats[f'{team}_team_id'] = raw_game['teams'][team]['id']
            # Calculate the basic stats
            for play in raw_game['plays']:
                if play['period'] <= 3:     # Exclude overtime stats
                    su.process_play_post_game(play, raw_game['teams'], game_stats)

            # Calculate CORSI
            for team in ['away', 'home']:
                for situation in SITUATIONS:
                    game_stats[f"{team}_CORSI_{situation}"] = (game_stats[f"{team}_G_{situation}"]
                                                               + game_stats[f"{team}_SOG_{situation}"]
                                                               + game_stats[f"{team}_SMISS_{situation}"])

                    # Add blocked shots (BLK is number of blocks of opposing team)
                    if team == "away":
                        game_stats[f"away_CORSI_{situation}"] += game_stats[f"home_BLK_{situation}"]
                    else:
                        game_stats[f"home_CORSI_{situation}"] += game_stats[f"away_BLK_{situation}"]

            # Calculate xG
            for team in ['away', 'home']:
                for situation in SITUATIONS:
                    # Filter subset of events
                    is_home = 1 if team == "home" else 0
                    subset = xg_pbp[(xg_pbp['game_id'] == raw_game['id']) & (xg_pbp['is_home'] == is_home)]
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
        df = df.sort_index(1)
        if save_to_csv:
            df.to_csv(self._data_path / "games_stats" / "post_game" / f"{season}-{season+1}.csv")
        else:
            return df

    def calculate_pre_game_stats(self, season, last_n_games=5, save_to_csv=False, verbose=0):
        """
        Create a dataset with pre-game stats (from post-game stats).

        Mean value of each stat from whole season and from last n games. With respect to home, away or both games.

        :param season:
        :param last_n_games:
        :return:
        """
        stats_post = pd.read_csv(
            self._data_path / "games_stats" / "post_game" / f"{season}-{season+1}.csv",
            index_col=0
        )
        stats_pre = []
        iterator = tqdm.tqdm(stats_post.index) if verbose else stats_post.index
        for game_id in iterator:
            team_stats = {
                "away": None,
                "home": None
            }
            # Get previous games of each team
            for team in ['away', 'home']:
                team_id = stats_post.loc[game_id, f"{team}_team_id"]

                # Get all previous away/home games of selected team
                prev_games = {
                    "away": None,
                    "home": None
                }
                for prev_games_type in ['away', 'home']:
                    prev_games[prev_games_type] = stats_post[
                        (stats_post.index < game_id) & (stats_post[f'{prev_games_type}_team_id'] == team_id)]

                # Get stats aggregates from previous away/home games of selected team
                team_stats[team] = self._get_stat_averages(
                    prev_games=prev_games, last_n_games=last_n_games
                )

            merged_stats = {}
            for team in ['away', 'home']:
                for stat in team_stats[team].keys():
                    merged_stats[f"{team}_{stat}"] = team_stats[team][stat]
            stats_pre.append(merged_stats)

        stats_pre = pd.DataFrame(stats_pre)
        stats_pre.index += 1
        if save_to_csv:
            stats_pre.to_csv(self._data_path / "games_stats" / "pre_game" / f"{season}-{season + 1}.csv")
        else:
            return stats_pre

    def _get_stat_averages(self, self2, prev_games, last_n_games=5):
        """
        Calculate mean values of basic stats from given history (previous games in the season) of a team.

        TODO: Add percentages (PP%, FO%, PK%, CORSI%, xG%,...)

        :param prev_games:
        :param last_n_games:
        :return:
        """
        basic_stats = ['G', 'SOG', 'SMISS', 'HIT', 'TAKE', 'GIVE', 'BLK', 'CORSI', 'xG']

        stats_aggr = {}     # Stats aggregates (mean values of past games)
        for active_team in ['away', 'home', 'both']:    # Previous away, home or all games of selected team
            opp_team = su.get_opp_team(active_team)
            for stat in basic_stats:
                for situation in SITUATIONS:
                    opp_situation = su.get_opp_situation(situation)
                    for last_games_flag in [False, True]:
                        for for_against in ['F', 'A']:
                            source_col = su.get_source_col_names(
                                for_against, situation, active_team, stat, opp_situation, opp_team
                            )

                            # Get per-game stats
                            if active_team == "both":
                                games_stats = pd.concat(
                                    [prev_games['away'][source_col[0]], prev_games['home'][source_col[1]]],
                                ).sort_index()
                            else:
                                games_stats = prev_games[active_team][source_col]

                            # Get name of the target column
                            if last_games_flag:
                                target_col = f"{stat}{for_against}_{situation}_{active_team}_last{last_n_games}"
                            else:
                                target_col = f"{stat}{for_against}_{situation}_{active_team}"

                            # Store the aggregate
                            if last_games_flag:
                                stats_aggr[target_col] = games_stats.tail(last_n_games).mean()
                            else:
                                stats_aggr[target_col] = games_stats.mean()
        return stats_aggr

    def get_whole_dataset(self):
        # Load inputs (x; pre-game stats)
        pre_game_stats_path = self._data_path / "games_stats" / "pre_game"
        dataframes = []
        for season in sorted(pre_game_stats_path.iterdir()):
            if season.suffix == ".csv":
                df = pd.read_csv(season, index_col=0)
                df['season'] = season.stem
                dataframes.append(df)
        x = pd.concat(dataframes).reset_index()

        # Load desired outputs (y; outcomes of the games)
        post_game_stats_path = self._data_path / "games_stats" / "post_game"
        dataframes = []
        for season in sorted(post_game_stats_path.iterdir()):
            if season.suffix == ".csv":
                df = pd.read_csv(season, index_col=0)
                df = df[['away_G_ALL', "home_G_ALL"]]
                df['season'] = season.stem
                dataframes.append(df)
        y = pd.concat(dataframes).reset_index()
        y["goal_diff"] = y['away_G_ALL'] - y['home_G_ALL']
        y['result'] = y['goal_diff'].apply(self.determine_winner)
        y = y[['index', 'season', 'result']]

        return x.drop(columns=['index', 'season']), y['result']
