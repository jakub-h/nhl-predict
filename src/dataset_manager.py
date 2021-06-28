import pandas as pd
from pathlib import Path
import pickle


class DatasetManager:
    """
    In-game stats: (*_away / *_home)
    G, SOG, SMISS, FO%, GIV, TAKE, PP, PIM, PPG, HIT, BLK,
    """
    def __init__(self, data_path):
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(f"'data_path' must be a string or instance of Path class. Not {type(data_path)}.")

    def calculate_in_game_stats(self, season):
        # Load games raw play-by-play
        with open(self._data_path / "games_raw" / f"{season}-{season+1}.pickle", 'rb') as f:
            games_raw = pickle.load(f)
        # Load xG play-by-play
        xg_pbp = pd.read_csv(self._data_path / "xg_pbp" / f"{season}-{season+1}.csv", index_col=0)
        print(xg_pbp.info())

        # return games_raw
