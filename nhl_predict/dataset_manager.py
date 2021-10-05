import pickle
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import tqdm
from sklearn.preprocessing import MinMaxScaler

import nhl_predict.stats.utils as su
from nhl_predict.xg.model_base import XGModel


class DatasetManager:
    """
    Class for managing datasets (stats):
    - create datasets
        - post-game (in-gmae stats; summary of each game)
        - pre-game (season stats; mean values for each team before the game starts)
    - split datasets into train and validation

    Helper functions are in `stats_utils.py`.
    """

    def __init__(self, data_path):
        self._SITUATIONS = ["ALL", "EV", "PP", "SH"]
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(
                f"'data_path' must be a string or instance of Path class. Not {type(data_path)}."
            )

    def calculate_post_game_stats(
        self, season: int, save_to_csv: bool = False
    ) -> Optional[pd.DataFrame]:
        """Create a dataset with post-game stats (actual stats from each game).

        - [away_ | home_] ... prefix
        - [_ALL | _EV | _PP | _SH] ... suffix
        - G, SOG, SMISS, HIT, TAKE, GIVE, FOW, PPO, PIM, BLK, CORSI, xG ... stats

        Parameters
        ----------
        season : int
            season to work with (e.g. 2011 for 2011-2012 season)
        save_to_csv : bool, optional
            if True, do not return anything and save the result to a file instead, by default False

        Returns
        -------
        Optional[pd.DataFrame]
            dataset with actual stats of both teams per game (summary of the game after it's finished).
            No overtime, only regulation.
        """
        # Load games raw play-by-play
        with open(
            self._data_path / "games_raw" / f"{season}-{season+1}.pickle", "rb"
        ) as f:
            games_raw = pickle.load(f)

        # Load xG play-by-play
        xg_pbp = pd.read_csv(
            self._data_path / "xg_pbp" / f"{season}-{season+1}.csv", index_col=0
        )

        # Load xG model
        xg_model = XGModel(
            self._data_path, model=self._data_path.parent / "models/xg_extra_trees_n10"
        )

        # Calculate xG predictions for all events (all games in the season)
        xg_pred = xg_model.predict(xg_pbp)
        xg_pbp["xg_pred"] = xg_pred

        # Define stats scheme
        stat_names = [
            "G",
            "SOG",
            "SMISS",
            "HIT",
            "TAKE",
            "GIVE",
            "FOW",
            "BLK",
            "CORSI",
            "xG",
        ]
        schema = {"away_team_id": None, "home_team_id": None}
        for team in ["away", "home"]:
            for situation in ["ALL", "EV", "PP", "SH"]:
                for stat in stat_names:
                    schema[f"{team}_{stat}_{situation}"] = 0
            schema[f"{team}_PPO"] = 0

        # Fill in stats
        games = []
        for raw_game in games_raw:
            game_stats = schema.copy()
            # Set team ids
            for team in ["away", "home"]:
                game_stats[f"{team}_team_id"] = raw_game["teams"][team]["id"]
            # Calculate the basic stats
            for play in raw_game["plays"]:
                if play["period"] <= 3:  # Exclude overtime stats
                    su.process_play_post_game(play, raw_game["teams"], game_stats)

            # Calculate CORSI
            for team in ["away", "home"]:
                for situation in ["ALL", "EV", "PP", "SH"]:
                    game_stats[f"{team}_CORSI_{situation}"] = (
                        game_stats[f"{team}_G_{situation}"]
                        + game_stats[f"{team}_SOG_{situation}"]
                        + game_stats[f"{team}_SMISS_{situation}"]
                    )

                    # Add blocked shots (BLK is number of blocks of opposing team)
                    if team == "away":
                        game_stats[f"away_CORSI_{situation}"] += game_stats[
                            f"home_BLK_{situation}"
                        ]
                    else:
                        game_stats[f"home_CORSI_{situation}"] += game_stats[
                            f"away_BLK_{situation}"
                        ]

            # Calculate xG
            for team in ["away", "home"]:
                for situation in self._SITUATIONS:
                    # Filter subset of events
                    is_home = 1 if team == "home" else 0
                    subset = xg_pbp[
                        (xg_pbp["game_id"] == raw_game["id"])
                        & (xg_pbp["is_home"] == is_home)
                    ]
                    if situation == "EV":
                        subset = subset[
                            subset["strength_active"] == subset["strength_opp"]
                        ]
                    elif situation == "PP":
                        subset = subset[
                            subset["strength_active"] > subset["strength_opp"]
                        ]
                    elif situation == "SH":
                        subset = subset[
                            subset["strength_active"] < subset["strength_opp"]
                        ]

                    # Calculate sum of xGs for events
                    xg_sum = 0 if subset.empty else subset["xg_pred"].sum()
                    game_stats[f"{team}_xG_{situation}"] = xg_sum

            games.append(game_stats)
        df = pd.DataFrame(games)
        df.index += 1
        df = df.sort_index(1)
        if save_to_csv:
            df.to_csv(
                self._data_path
                / "games_stats"
                / "post_game"
                / f"{season}-{season+1}.csv"
            )
        else:
            return df

    def calculate_pre_game_stats(
        self,
        season: int,
        last_n_games: int = 3,
        save_to_csv: bool = False,
        verbose: int = 0,
    ) -> Optional[pd.DataFrame]:
        """Create a dataset with pre-game stats (from post-game stats).

        Mean value of each stat from whole season and from last n games. With respect to home, away or both games.

        Parameters
        ----------
        season : int
            season to work with (e.g. 2011 for 2011-2012 season)
        last_n_games : int, optional
            how many most recent games to be treated as 'recent shape' of the team, by default 3
        save_to_csv : bool, optional
            if True, do not return anything and save the result to a file instead, by default False
        verbose : int, optional
            verbosity level, by default 0

        Returns
        -------
        Optional[pd.DataFrame]
            dataset with mean values of stats from previous games of each team per game
        """
        stats_post = pd.read_csv(
            self._data_path / "games_stats" / "post_game" / f"{season}-{season+1}.csv",
            index_col=0,
        )
        stats_pre = []
        iterator = tqdm.tqdm(stats_post.index) if verbose else stats_post.index
        for game_id in iterator:
            team_stats = {"away": None, "home": None}
            # Get previous games of each team
            for team in ["away", "home"]:
                team_id = stats_post.loc[game_id, f"{team}_team_id"]

                # Get all previous away/home games of selected team
                prev_games = {"away": None, "home": None}
                for prev_games_type in ["away", "home"]:
                    prev_games[prev_games_type] = stats_post[
                        (stats_post.index < game_id)
                        & (stats_post[f"{prev_games_type}_team_id"] == team_id)
                    ]

                # Get stats aggregates from previous away/home games of selected team
                team_stats[team] = self._get_stat_averages(
                    prev_games=prev_games, last_n_games=last_n_games
                )

            merged_stats = {}
            for team in ["away", "home"]:
                for stat in team_stats[team].keys():
                    merged_stats[f"{team}_{stat}"] = team_stats[team][stat]
            stats_pre.append(merged_stats)

        stats_pre = pd.DataFrame(stats_pre)
        stats_pre.index += 1
        if save_to_csv:
            stats_pre.to_csv(
                self._data_path
                / "games_stats"
                / "pre_game"
                / f"{season}-{season + 1}.csv"
            )
        else:
            return stats_pre

    def get_stat_averages(self, prev_games: dict, last_n_games: int = 3) -> dict:
        """Calculate mean values of basic stats from given history (previous games in the season) of a team.

        TODO: Add percentages (PP%, FO%, PK%, CORSI%, xG%,...)

        Parameters
        ----------
        prev_games : dict
            previous games of the team in current season (divided into away and home games)
        last_n_games : int, optional
            how many most recent games to be treated as 'recent shape' of the team, by default 3

        Returns
        -------
        dict
            mean values of the stats from previous games of the team (before current game starts)
        """
        basic_stats = ["G", "SOG", "SMISS", "HIT", "TAKE", "GIVE", "BLK", "CORSI", "xG"]

        stats_aggr = {}  # Stats aggregates (mean values of past games)
        for active_team in [
            "away",
            "home",
            "both",
        ]:  # Previous away, home or all games of selected team
            opp_team = su.get_opp_team(active_team)
            for stat in basic_stats:
                for situation in self._SITUATIONS:
                    opp_situation = su.get_opp_situation(situation)
                    for last_games_flag in [False, True]:
                        for for_against in ["F", "A"]:
                            source_col = su.get_source_col_names(
                                for_against,
                                situation,
                                active_team,
                                stat,
                                opp_situation,
                                opp_team,
                            )

                            # Get per-game stats
                            if active_team == "both":
                                games_stats = pd.concat(
                                    [
                                        prev_games["away"][source_col[0]],
                                        prev_games["home"][source_col[1]],
                                    ],
                                ).sort_index()
                            else:
                                games_stats = prev_games[active_team][source_col]

                            # Get name of the target column
                            if last_games_flag:
                                target_col = f"{stat}{for_against}_{situation}_{active_team}_last{last_n_games}"
                            else:
                                target_col = (
                                    f"{stat}{for_against}_{situation}_{active_team}"
                                )

                            # Store the aggregate
                            if last_games_flag:
                                stats_aggr[target_col] = games_stats.tail(
                                    last_n_games
                                ).mean()
                            else:
                                stats_aggr[target_col] = games_stats.mean()
        return stats_aggr

    def get_sample_data(self) -> Tuple[pd.DataFrame]:
        x_sample, y_sample = self._load_dataset([2011, 2018])
        return x_sample, y_sample

    def cross_validation(
        self, num_train_seasons: int = 3, num_val_seasons: int = 1, one_hot: bool = True
    ) -> Tuple[pd.DataFrame]:
        """Task-specific cross validation (generator).

        In one fold, `train_seasons` consecutive seasons are used for training and following `val_seasons` are used for
        validation.

        Parameters
        ----------
        num_train_seasons : int, optional
            number of train seasons in each fold, by default 3
        num_val_seasons : int, optional
            number of validation seasons in each fold, by default 1
        one_hot : bool, optional
            if True, encode labels (y_true) to one-hot encoding, by default True

        Yields
        -------
        Tuple[pd.DataFrame]
            x_train, x_val, y_train, y_val (scaled, with NaNs, `y_train` and `y_val` one-hot encoded if `one_hot`)
        """
        seasons = [
            int(x.stem.split("-")[0])
            for x in sorted((self._data_path / "games_stats" / "pre_game").iterdir())
            if x.suffix == ".csv"
        ]
        print(seasons)

        i = 0
        train_seasons = seasons[:num_train_seasons]
        val_seasons = seasons[num_train_seasons : num_train_seasons + num_val_seasons]
        while len(val_seasons) == num_val_seasons and val_seasons[-1] <= seasons[-1]:
            x_train, y_train = self._load_dataset(train_seasons)
            x_val, y_val = self._load_dataset(val_seasons)

            scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
            x_train = pd.DataFrame(
                scaler.transform(x_train), index=x_train.index, columns=x_train.columns
            )
            x_val = pd.DataFrame(
                scaler.transform(x_val), index=x_val.index, columns=x_val.columns
            )

            if one_hot:
                y_train = pd.get_dummies(y_train)
                y_val = pd.get_dummies(y_val)

            yield x_train, x_val, y_train, y_val

            i += 1
            train_seasons = seasons[i : i + num_train_seasons]
            val_seasons = seasons[
                i + num_train_seasons : i + num_train_seasons + num_val_seasons
            ]

    def _load_dataset(self, seasons: Iterable[int]) -> Tuple[pd.DataFrame]:
        """Load pre-game dataset combined from `seasons` list.

        Parameters
        ----------
        seasons : Iterable[int]
            list of seasons for the dataset (e.g. [2011, 2012, 2013])

        Returns
        -------
        Tuple[pd.DataFrame]
            (x, y) - x: inputs (pre-game stats per game)
                   - y: outputs (outcome [away team win, draw, home team win] of each game)
        """
        inputs = []
        outputs = []
        for season in seasons:
            # Inputs (x; pre-game stats)
            season_path = (
                self._data_path
                / "games_stats"
                / "pre_game"
                / f"{season}-{season+1}.csv"
            )
            df = pd.read_csv(season_path, index_col=0)
            df["season"] = f"{season}-{season+1}"
            inputs.append(df)

            # Outputs (y; outcomes of the games)
            season_path = (
                self._data_path
                / "games_stats"
                / "post_game"
                / f"{season}-{season+1}.csv"
            )
            df = pd.read_csv(season_path, index_col=0)
            df = df[["away_G_ALL", "home_G_ALL"]]
            df["season"] = f"{season}-{season+1}"
            outputs.append(df)
        x = pd.concat(inputs).reset_index()
        y = pd.concat(outputs).reset_index()

        # Basic preprocessing
        season_col = x.pop("season")
        x.insert(0, "season", season_col)
        x = x.rename(columns={"index": "game_id"})
        x = x.set_index(["season", "game_id"])

        y["goal_diff"] = y["away_G_ALL"] - y["home_G_ALL"]
        y["result"] = y["goal_diff"].apply(su.determine_winner)
        y = y.rename(columns={"index": "game_id"})
        y = y[["season", "game_id", "result"]]
        y = y.set_index(["season", "game_id"])
        return x, y
