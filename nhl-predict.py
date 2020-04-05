from dataset_manager import DatasetManager
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import scipy.stats
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, ParameterGrid
import colorlover as cl
import time



def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    n = len(a)
    m = np.mean(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h




def rf_grid_search_CV(dm, clf, params_grid):
    print("## Grid Search CV ##: Random Forest")
    for dt in [False]:
        for dc in [False]:
            for scale in ['std']:
                for params in ParameterGrid(params_grid):
                    start = time.time()
                    results_train = []
                    results_test = []
                    print(params)
                    print("'dummy_teams': {}, 'dummy_coaches': {}, 'scale': {} ---> ".format(dt, dc, scale), end='', flush=True)
                    for fold_i in range(5):
                        x_train, x_test, y_train, y_test = dm.get_fold(fold_i, dt, dc, scale)
                        clf = RandomForestClassifier(**params, n_jobs=-1)
                        clf.fit(x_train, y_train)
                        results_train.append(clf.score(x_train, y_train)*100)
                        results_test.append(clf.score(x_test, y_test)*100)
                    m_train, ci_train = mean_confidence_interval(results_train)
                    m_test, ci_test = mean_confidence_interval(results_test)
                    print("train: {:.2f}% +-{:.2f}%; test: {:.2f}% +-{:.2f}%".format(m_train, ci_train, m_test, ci_test))
                    end = time.time()
                    print(25*"-", "done in {:.2f} s".format(end-start), 25*"-")


if __name__ == "__main__":
    dm = DatasetManager(games_hr_fn=None)
    clf = RandomForestClassifier()
    params = {
        'n_estimators': [50, 400, 800],
        'criterion': ['gini', 'entropy'],
        'max_depth': [100, 400, None],
        'min_samples_split': [2, 8, 32],
        'min_samples_leaf': [1, 4, 16],
        'max_features': [None, 'sqrt', 'log2'],
    }
    rf_grid_search_CV(dm, None, params)
