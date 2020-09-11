from dataset_manager_v1 import DatasetManager
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


if __name__ == "__main__":
    dm = DatasetManager()
    for test_season in range(2015, 2020):
        for n_games in [0, 100, 200]:
            for last_n in [2, 5, 10]:
                for dummy in [False, True]:
                    dm.create_seasonal_split(test_season, n_games, 2, last_n, dummy)
    