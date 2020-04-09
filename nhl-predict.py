from dataset_manager import DatasetManager
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import scipy.stats
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
import colorlover as cl
import time



def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    n = len(a)
    m = np.mean(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def plot_conf_matrix(conf_matrix, dataset_name):
    fig = ff.create_annotated_heatmap (
        conf_matrix,
        colorscale='Blues',
        x = ['away', 'draw', 'home'],
        y = ['away', 'draw', 'home'],
        showscale=True,
        annotation_text = np.around(conf_matrix, decimals=2),
        hoverinfo='z')
    fig.update_layout(
        title = "Confusion matrix - '{}'".format(dataset_name),
        xaxis_title = {
            'text': "predicted class",
            'font': {'size': 10} 
        },
        yaxis_title = {
            'text': "true class",
            'font': {'size': 10} 
        },
        xaxis = {
            'linecolor': 'black',
            'mirror': True
        },
        yaxis = {
            'linecolor': 'black',
            'mirror': True
        },
        width = 500,
        height = 500
    )
    fig.show()


def erf_grid_search_CV(dm, param_grid):
    print("## Grid Search CV ##: Extreme Random Forest")
    results = {'dummy_teams': [], 'dummy_coaches': [], 'scale': []}
    for param_name in ParameterGrid(param_grid)[0].keys():
        results[param_name] = []
    results['acc_train'] = []
    results['ci_train'] = []
    results['acc_test'] = []
    results['ci_test'] = []
    results['time'] = []
    for scale in [None, 'std']:
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
                clf = ExtraTreesClassifier(**params, n_jobs=15)
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
    results.to_csv("results/extra_trees_gridsearch.csv")


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


def seasonal_grid_search(dm, params):
    print("## Grid Search CV ##: Random Forest (Seasonal)")
    results = {
        'test_season': [],
        'first_games': [],
        'train_seasons': [],
        'acc_train': [],
        'acc_test': []
    }
    for test_season in [2017, 2018, 2019]:
        for train_num in range(1, test_season-2015):
            train_seasons = []
            for i_train in range(train_num):
                train_seasons.append(test_season-1-i_train)
            for first_games in [100, 200, 400, 800]:
                print("'test_season': {}; 'first_n_games_to_train': {}; 'trian_seasons': {} ---> ".format(
                    test_season, first_games, train_seasons), end='', flush=True)
                x_train, x_test, y_train, y_test = dm.get_seasonal_split(test_season, first_games, train_seasons)
                start = time.time()
                clf = RandomForestClassifier(**params).fit(x_train, y_train)
                results['test_season'].append(test_season)
                results['first_games'].append(first_games)
                results['train_seasons'].append(train_seasons)
                acc_train = clf.score(x_train, y_train)*100
                acc_test = clf.score(x_test, y_test)*100
                results['acc_train'].append(acc_train)
                results['acc_test'].append(acc_test)
                end = time.time()
                print("train: {:.2f}%; test: {:.2f}%".format(acc_train, acc_test))
                print(25*"-", "done in {:.2f} s".format(end-start), 25*"-")
    results = pd.DataFrame.from_dict(results)
    results.to_csv("results/rf_seasonal_gs.csv")



def only_sure():
    dm = DatasetManager(base_dataset_fn=None)
    x_train, x_test, y_train, y_test = dm.get_seasonal_split(2019, 200, [2016, 2017, 2018])
    clf = RandomForestClassifier(n_estimators=400, max_depth=500, min_samples_split=8, verbose=1,
                                 max_features=None, criterion='entropy', n_jobs=-1).fit(x_train, y_train)
    target_names = ['away', 'draw', 'home']
    y_pred = clf.predict(x_train)
    print(classification_report(y_train, y_pred, target_names=target_names))
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=target_names))

    probs = clf.predict_proba(x_test)
    y_pred = pd.Series(clf.predict(x_test), index=y_test.index)
    y_sure = []
    for prob in probs:
        if sorted(prob, reverse=True)[0] > 0.7:
            y_sure.append(True)
        else:
            y_sure.append(False)
    y_sure = pd.Series(y_sure, index=y_test.index)
    results = pd.DataFrame()
    results['true'] = y_test
    results['predict'] = y_pred
    results['sure'] = y_sure
    y_true = results.loc[results['sure'] == True, 'true']
    y_pred = results.loc[results['sure'] == True, 'predict']
    print(classification_report(y_true, y_pred, target_names=['away', 'draw', 'home']))



def seasonal_crossval(params, first_games, num_of_training_seasons, probs=False):
    dm = DatasetManager(base_dataset_fn=None, player_names_fn=None, team_names_fn=None)
    print("## Seasonal CV ##: RandomForest ({})".format(params))
    y_true = []
    y_pred = []
    test_seasons = [x for x in range(2016, 2020)]
    if num_of_training_seasons == 2:
        test_seasons.append(2015)
    for test_season in sorted(test_seasons):
        train_seasons = sorted([test_season-1-x for x in range(num_of_training_seasons)])
        print("test season: {} (first {} games to train); train seasons: {} ---> ".format(test_season, first_games, train_seasons), end='', flush=True)
        start = time.time()
        x_train, x_test, y_train, y_test = dm.get_seasonal_split(test_season, first_games, train_seasons)
        clf = RandomForestClassifier(**params).fit(x_train, y_train)
        end = time.time()
        y_true.append(y_test.rename('true'))
        if probs:
            y_pred.append(pd.DataFrame(clf.predict_proba(x_test), index=y_test.index, columns=['away', 'draw', 'home']))
        else:
            y_pred.append(pd.Series(clf.predict(x_test), index=y_test.index, name='pred'))
        print('done in {:.2f} s'.format(end - start))
    return(y_true, y_pred)




if __name__ == "__main__":
    params = {
        'criterion': 'entropy',
        'n_estimators': 200,
        'max_depth': 500,
        'max_features': None,
        'min_samples_split': 8,
        'min_samples_leaf': 1,
        'n_jobs': 3
    }
    y_true, y_pred = seasonal_crossval(params, 200, 3)
    acc = []
    prec = []
    for y_t, y_p in zip(y_true, y_pred):
        acc.append(accuracy_score(y_t, y_p))
        prec.append(precision_score(y_t, y_p, average='weighted'))
    acc_m, acc_ci = mean_confidence_interval(acc)
    prec_m, prec_ci = mean_confidence_interval(prec)
    print("--> accuracy: {:.2f}% +-{:.2f}%".format(acc_m*100, acc_ci*100))
    print("--> precision: {:.2f}% +-{:.2f}%".format(prec_m*100, prec_ci*100))
    y_true = pd.concat(y_true)
    y_pred = pd.concat(y_pred)
    print("OVERALL:")
    print(classification_report(y_true, y_pred, target_names=['away', 'draw', 'home']))
