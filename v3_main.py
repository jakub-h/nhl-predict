from dataset_manager_v3 import DatasetManager
import numpy as np
import scipy


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    n = len(a)
    m = np.mean(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h



if __name__ == "__main__":
    dm = DatasetManager()
    first_games = [0]
    form_lengths = [5]
    for test_season in range(2017, 2020):
        for first_game in first_games:
            for form_length in form_lengths:
                dm.create_seasonal_split(test_season, first_game, 4, form_length, False, False, False)