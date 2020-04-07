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




def rf_grid_search_CV(dm, param_grid):
    print("## Grid Search CV ##: Random Forest")
    results = {'dummy_teams': [], 'dummy_coaches': [], 'scale': []}
    for param_name in ParameterGrid(param_grid)[0].keys():
        results[param_name] = []
    results['acc_train'] = []
    results['ci_train'] = []
    results['acc_test'] = []
    results['ci_test'] = []
    results['time'] = []
    for scale in ['std']:
        for params in ParameterGrid(param_grid):
            results['dummy_teams'].append(False)
            results['dummy_coaches'].append(False)
            results['scale'].append(scale)
            for param_name in params.keys():
                results[param_name].append(params[param_name])
            results_train = []
            results_test = []
            print(params)
            print("'dummy_teams': {}, 'dummy_coaches': {}, 'scale': {} ---> ".format(False, False, scale), end='', flush=True)
            start = time.time()
            for fold_i in range(5):
                x_train, x_test, y_train, y_test = dm.get_fold(fold_i, False, False, scale)
                clf = RandomForestClassifier(**params, n_jobs=15)
                clf.fit(x_train, y_train)
                results_train.append(clf.score(x_train, y_train)*100)
                results_test.append(clf.score(x_test, y_test)*100)
            m_train, ci_train = mean_confidence_interval(results_train)
            m_test, ci_test = mean_confidence_interval(results_test)
            end = time.time()
            results['acc_train'].append(m_train)
            results['ci_train'].append(ci_train)
            results['acc_test'].append(m_test)
            results['ci_test'].append(ci_test)
            results['time'].append(np.round(end-start, 2))
            print("train: {:.2f}% +-{:.2f}%; test: {:.2f}% +-{:.2f}%".format(m_train, ci_train, m_test, ci_test))
            print(25*"-", "done in {:.2f} s".format(end-start), 25*"-")
    results = pd.DataFrame.from_dict(results)
    results.to_csv("results/random_forest_gridsearch_2.csv")


if __name__ == "__main__":
    dm = DatasetManager(games_hr_fn=None)
    clf = RandomForestClassifier()
    params = {
        'n_estimators': [50],
        'criterion': ['entropy'],
        'max_depth': [300, 500],
        'min_samples_split': [2, 4, 8, 16],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6],
        'max_features': [None],
    }
    rf_grid_search_CV(dm, params)
