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
    for minmax in [True, False]:
        dm.create_seasonal_split(2019, 0, 6, 5, False, False, minmax)