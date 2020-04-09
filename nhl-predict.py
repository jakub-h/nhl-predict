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
import json



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



def seasonal_crossval(params, first_games, num_of_training_seasons):
    dm = DatasetManager(base_dataset_fn=None, player_names_fn=None, team_names_fn=None)
    print("## Seasonal CV ##: RandomForest ({})".format(params))
    y_true = []
    y_pred = []
    y_probs = []
    test_seasons = [x for x in range(2016, 2020)]
    if num_of_training_seasons < 3:
        test_seasons.append(2015)
    if num_of_training_seasons < 2:
        test_seasons.append(2014)
    for test_season in sorted(test_seasons):
        train_seasons = sorted([test_season-1-x for x in range(num_of_training_seasons)])
        print("test season: {} (first {} games to train); train seasons: {} ---> ".format(test_season, first_games, train_seasons), end='', flush=True)
        start = time.time()
        x_train, x_test, y_train, y_test = dm.get_seasonal_split(test_season, first_games, train_seasons)
        clf = RandomForestClassifier(**params,
                                     n_jobs=20, criterion='entropy', min_samples_leaf=1,
                                     min_samples_split=8, max_features=None).fit(x_train, y_train)
        end = time.time()
        y_true.append(y_test.rename('true'))
        y_probs.append(pd.DataFrame(clf.predict_proba(x_test), index=y_test.index, columns=['away', 'draw', 'home']))
        y_pred.append(pd.Series(clf.predict(x_test), index=y_test.index, name='pred'))
        print('done in {:.2f} s'.format(end - start))
    return y_true, y_pred, y_probs


def seasonal_grid_search(clf_param_grid, first_games, nums_of_train_seasons, conf_factors):
    print("## Grid Search CV ##: Random Forest (Seasonal)")
    results = {
        'n_estimators': [],
        'max_depth': [],
        'first_games': [],
        'num_train_seasons': [],
        'cv_acc_mean': [],
        'cv_acc_ci': [],
        'cv_prec_mean': [],
        'cv_prec_ci': [],
        'away_precision': [],
        'away_recall': [],
        'away_support': [],
        'draw_precision': [],
        'draw_recall': [],
        'draw_support': [],
        'home_precision': [],
        'home_recall': [],
        'home_support': [],
        'ovr_acc': [],
        'confidence_factor': [],
        'sure_away_precision': [],
        'sure_away_recall': [],
        'sure_away_support': [],
        'sure_draw_precision': [],
        'sure_draw_recall': [],
        'sure_draw_support': [],
        'sure_home_precision': [],
        'sure_home_recall': [],
        'sure_home_support': [],
        'sure_ovr_acc': [],
        'train_time': [],
    }
    for params in ParameterGrid(clf_param_grid):
        for n_games in first_games:
            for n_seasons in nums_of_train_seasons:
                for param_name in params.keys():
                    for _ in range(len(conf_factors)):
                        results[param_name].append(params[param_name])
                for _ in range(len(conf_factors)):
                    results['first_games'].append(n_games)
                    results['num_train_seasons'].append(n_seasons)
                # Get predictions
                start = time.time()
                y_true, y_pred, y_probs = seasonal_crossval(params, n_games, n_seasons)
                end = time.time()
                for _ in range(len(conf_factors)):
                    results['train_time'].append(end - start)
                # Cross validation estimations (seasons separately -> mean + confidence intervals)
                acc = []
                prec = []
                for y_t, y_p in zip(y_true, y_pred):
                    acc.append(accuracy_score(y_t, y_p))
                    prec.append(precision_score(y_t, y_p, average='weighted'))
                acc_m, acc_ci = mean_confidence_interval(acc)
                prec_m, prec_ci = mean_confidence_interval(prec)
                for _ in range(len(conf_factors)):
                    results['cv_acc_mean'].append(acc_m)
                    results['cv_acc_ci'].append(acc_ci)
                    results['cv_prec_mean'].append(prec_m)
                    results['cv_prec_ci'].append(prec_ci)
                # Overall performance (without filtering unsure predictions)
                class_names = ['away', 'draw', 'home']
                y_true = pd.concat(y_true)
                y_pred = pd.concat(y_pred)
                cl_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
                for _ in range(len(conf_factors)):
                    results['ovr_acc'].append(cl_rep['accuracy'])
                for cl in ['away', 'draw', 'home']:
                    for met in ['precision', 'recall', 'support']:
                        for _ in range(len(conf_factors)):
                            results['{}_{}'.format(cl, met)].append(cl_rep[cl][met])
                # Filtering unsure predictions
                y_probs = pd.concat(y_probs)
                for conf_fac in conf_factors:
                    results['confidence_factor'].append(conf_fac)
                    y_sure = []
                    for _, row in y_probs.iterrows():
                        if sorted(row, reverse=True)[0] > conf_fac:
                            y_sure.append(True)
                        else:
                            y_sure.append(False)
                    y_sure = pd.Series(y_sure, index=y_true.index)
                    y_true_sure = y_true.loc[y_sure == True]
                    y_pred_sure = y_pred.loc[y_sure == True]
                    cl_rep_sure = classification_report(y_true_sure, y_pred_sure, target_names=class_names, output_dict=True)
                    results['sure_ovr_acc'].append(cl_rep_sure['accuracy'])
                    for cl in ['away', 'draw', 'home']:
                        for met in ['precision', 'recall', 'support']:
                            results['sure_{}_{}'.format(cl, met)].append(cl_rep_sure[cl][met])
                with open('results/seasonal_gridsearch_backup.json', 'w') as f:
                    json.dump(results, f)
    results = pd.DataFrame.from_dict(results)
    results.to_csv("results/seasonal_gridsearch.csv")  



if __name__ == "__main__":
    dm = DatasetManager()
    param_grid = {
        'n_estimators': [50, 200, 400, 600],
        'max_depth': [20, 300, 500, 700]
    }
    first_games = [0, 100, 200, 300, 400]
    train_seasons_num = [1, 2, 3]
    confidence_factors = [0.5, 0.7, 0.9]
    seasonal_grid_search(param_grid, first_games, train_seasons_num, confidence_factors)
    

