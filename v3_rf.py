from dataset_manager_v3 import DatasetManager
import json
import os
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    n = len(a)
    m = np.mean(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h



def seasonal_crossval(params, first_games, num_of_training_seasons, form_length, dummy, home_rel, minmax, n_jobs, use_class_weights):
    dm = DatasetManager(base_dataset_fn=None)
    print("## Seasonal CV ##: RandomForest ({})".format(params))
    print("## form: {}; dummy: {}; home_rel: {}; minmax: {}; class_weights: {}".format(form_length, dummy, home_rel, minmax, use_class_weights))
    y_true = []
    y_pred = []
    y_probs = []
    for test_season in range(2017, 2020):
        print("test season: {} (first {} games to train); train seasons: {} ---> ".format(test_season, first_games, num_of_training_seasons), end='', flush=True)
        start = time.time()
        x_train, x_test, y_train, y_test = dm.get_seasonal_split(test_season, first_games, num_of_training_seasons, form_length, dummy, home_rel, minmax)
        if use_class_weights:
            class_weights = y_train.value_counts().to_dict()
        else:
            class_weights = None
        clf = RandomForestClassifier(**params, class_weight=class_weights, n_jobs=n_jobs).fit(x_train, y_train)
        end = time.time()
        y_true.append(y_test.rename('true'))
        y_probs.append(pd.DataFrame(clf.predict_proba(x_test), index=y_test.index, columns=['away', 'draw', 'home']))
        y_pred.append(pd.Series(clf.predict(x_test), index=y_test.index, name='pred'))
        print('done in {:.2f} s'.format(end - start))
    return y_true, y_pred, y_probs
    

def seasonal_grid_search(clf_param_grid, first_games, conf_factors, form_lengths):
    print("## Grid Search CV ##: Random Forest (Seasonal)")
    results = {
        'n_estimators': [],
        'max_depth': [],
        'min_samples_leaf': [],
        'min_samples_split': [],
        'max_features': [],
        'criterion': [],
        'class_weights': [],
        'first_games': [],
        'form_length': [],
        'dummy': [],
        'home_rel': [],
        'minmax': [],
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
            for class_weights in [False]:
                for form_length in form_lengths:
                    for dummy in [False]:
                        for home_rel in [False]:
                            for minmax in [False]:
                                for param_name in params.keys():
                                    for _ in range(len(conf_factors)):
                                        results[param_name].append(params[param_name])
                                for _ in range(len(conf_factors)):
                                    results['class_weights'].append(class_weights)
                                    results['first_games'].append(n_games)
                                    results['form_length'].append(form_length)
                                    results['dummy'].append(dummy)
                                    results['home_rel'].append(home_rel)
                                    results['minmax'].append(minmax)
                                # Get predictions
                                start = time.time()
                                y_true, y_pred, y_probs = seasonal_crossval(params, n_games, 2, form_length, dummy, home_rel, minmax, -1, class_weights)
                                end = time.time()
                                for _ in range(len(conf_factors)):
                                    results['train_time'].append(end - start)
                                # Cross validation estimations (seasons separately -> mean + confidence intervals)
                                acc = []
                                prec = []
                                for y_t, y_p in zip(y_true, y_pred):
                                    acc.append(accuracy_score(y_t, y_p))
                                    prec.append(precision_score(y_t, y_p, average='weighted', zero_division=0))
                                acc_m, acc_ci = mean_confidence_interval(acc)
                                prec_m, prec_ci = mean_confidence_interval(prec)
                                for _ in range(len(conf_factors)):
                                    results['cv_acc_mean'].append(acc_m)
                                    results['cv_acc_ci'].append(acc_ci)
                                    results['cv_prec_mean'].append(prec_m)
                                    results['cv_prec_ci'].append(prec_ci)
                                # Overall performance (without filtering unsure predictions)
                                y_true = pd.concat(y_true)
                                y_pred = pd.concat(y_pred)
                                cl_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
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
                                    if not y_pred_sure.empty:
                                        cl_rep_sure = classification_report(y_true_sure, y_pred_sure, output_dict=True, zero_division=0)
                                    else:
                                        cl_rep_sure = None
                                    if cl_rep_sure is not None:
                                        results['sure_ovr_acc'].append(cl_rep_sure['accuracy'])
                                    else:
                                        results['sure_ovr_acc'].append(0)
                                    for cl in ['away', 'draw', 'home']:
                                        for met in ['precision', 'recall', 'support']:
                                            if cl_rep_sure is not None and cl in cl_rep_sure:
                                                results['sure_{}_{}'.format(cl, met)].append(cl_rep_sure[cl][met])
                                            else:
                                                results['sure_{}_{}'.format(cl, met)].append(0)
                                with open('results/v3/rf_seasonal_gridsearch_backup_2.json', 'w') as f:
                                    json.dump(results, f, indent=4)
    results = pd.DataFrame.from_dict(results) 
    results.to_csv("results/v3/rf_seasonal_gridsearch_2.csv")



if __name__ == "__main__":
    params = {
        'n_estimators': 1000,
        'criterion': 'entropy',
        'max_depth': None,
        'min_samples_leaf': 8,
    }
    confidence_factor = 0.5
    y_true, y_pred, y_probs = seasonal_crossval(params, first_games=0, num_of_training_seasons=6, 
                                                form_length=5, dummy=False, home_rel=False, minmax=False,
                                                n_jobs=3, use_class_weights=False)
    y_true = pd.concat(y_true)
    y_pred = pd.concat(y_pred)
    y_probs = pd.concat(y_probs)
    print(10*"=", 'y_true', 10*'=')
    print(y_true.value_counts())
    print(10*"=", 'y_pred', 10*'=')
    print(y_pred.value_counts())
    print(classification_report(y_true, y_pred, zero_division=0, digits=4))
    y_sure = []
    for _, row in y_probs.iterrows():
        if sorted(row)[-1] > confidence_factor:
            y_sure.append(True)
        else:
            y_sure.append(False)
    y_sure = pd.Series(y_sure, index=y_true.index)
    y_true_sure = y_true.loc[y_sure == True]
    y_pred_sure = y_pred.loc[y_sure == True]
    print(10*"=", 'y_true_sure (conf_f = {:.2f})'.format(confidence_factor), 10*'=')
    print(y_true_sure.value_counts())
    print(10*"=", 'y_pred_sure (conf_f = {:.2f})'.format(confidence_factor), 10*'=')
    print(y_pred_sure.value_counts())
    print(classification_report(y_true_sure, y_pred_sure, zero_division=0, digits=4))
    
    
    
    
    '''      
    param_grid = {
        'n_estimators': [50],
        'max_depth': [10, 50, 300, None],
        'min_samples_leaf': [1, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt'],
        'criterion': ['entropy']
    }
    first_games = [0, 100, 300]
    confidence_factors = [0.5, 0.55, 0.6]
    form_lengths = [1, 3, 5, 10]
    seasonal_grid_search(param_grid, first_games, confidence_factors, form_lengths)
    '''

    
    
