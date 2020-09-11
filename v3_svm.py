from dataset_manager_v3 import DatasetManager
from sklearn.svm import SVC
import pandas as pd
import time
import json
from sklearn.metrics import accuracy_score, precision_score, classification_report
from v3_main import mean_confidence_interval
from sklearn.model_selection import ParameterGrid


def seasonal_crossval(params, first_games, num_of_training_seasons, form_length, dummy, home_rel, minmax):
    dm = DatasetManager(base_dataset_fn=None)
    print("## Seasonal CV ##: SVM ({})".format(params))
    print("## form: {}; dummy: {}; home_rel: {}; minmax: {}".format(form_length, dummy, home_rel, minmax))
    y_true = []
    y_pred = []
    #y_probs = []
    for test_season in range(2015, 2020):
        print("test season: {} (first {} games to train); train seasons: {} ---> ".format(test_season, first_games, num_of_training_seasons), end='', flush=True)
        start = time.time()
        x_train, x_test, y_train, y_test = dm.get_seasonal_split(test_season, first_games, num_of_training_seasons, form_length, dummy, home_rel, minmax)
        clf = SVC(**params).fit(x_train, y_train)
        end = time.time()
        y_true.append(y_test.rename('true'))
        #y_probs.append(pd.DataFrame(clf.predict_proba(x_test), index=y_test.index, columns=['away', 'draw', 'home']))
        y_pred.append(pd.Series(clf.predict(x_test), index=y_test.index, name='pred'))
        print('done in {:.2f} s'.format(end - start))
    return y_true, y_pred#, y_probs
    

def seasonal_grid_search(clf_param_grid, form_lengths):
    print("## Grid Search CV ##: SVM (Seasonal)")
    results = {
        'kernel': [],
        'C': [],
        'coef0': [],
        'gamma': [],
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
        'train_time': [],
    }
    for params in ParameterGrid(clf_param_grid):
        for form_length in form_lengths:
            for dummy in [False, True]:
                for home_rel in [False, True]:
                    for minmax in [False, True]:
                        for param_name in params.keys():
                            results[param_name].append(params[param_name])
                        results['form_length'].append(form_length)
                        results['dummy'].append(dummy)
                        results['home_rel'].append(home_rel)
                        results['minmax'].append(minmax)
                        # Get predictions
                        start = time.time()
                        y_true, y_pred = seasonal_crossval(params, 0, 2, form_length, dummy, home_rel, minmax)
                        end = time.time()
                        results['train_time'].append(end - start)
                        # Cross validation estimations (seasons separately -> mean + confidence intervals)
                        acc = []
                        prec = []
                        for y_t, y_p in zip(y_true, y_pred):
                            acc.append(accuracy_score(y_t, y_p))
                            prec.append(precision_score(y_t, y_p, average='weighted', zero_division=0))
                        acc_m, acc_ci = mean_confidence_interval(acc)
                        prec_m, prec_ci = mean_confidence_interval(prec)
                        results['cv_acc_mean'].append(acc_m)
                        results['cv_acc_ci'].append(acc_ci)
                        results['cv_prec_mean'].append(prec_m)
                        results['cv_prec_ci'].append(prec_ci)
                        # Overall performance (without filtering unsure predictions)
                        y_true = pd.concat(y_true)
                        y_pred = pd.concat(y_pred)
                        cl_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        results['ovr_acc'].append(cl_rep['accuracy'])
                        for cl in ['away', 'draw', 'home']:
                            for met in ['precision', 'recall', 'support']:
                                results['{}_{}'.format(cl, met)].append(cl_rep[cl][met])
                        with open('results/v3/svm_seasonal_gridsearch_backup.json', 'w') as f:
                            json.dump(results, f, indent=4)
    results = pd.DataFrame.from_dict(results) 
    results.to_csv("results/v3/svm_seasonal_gridsearch.csv")


if __name__ == "__main__":
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'coef0': [0, -1, 1]
    }
    form_lengths = [5]
    seasonal_grid_search(param_grid, form_lengths)
    '''
    param_grid = {
        'n_estimators': [50, 200],
        'max_depth': [50, 300, None],
        'min_samples_leaf': [1, 5, 10],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', None],
        'criterion': ['gini', 'entropy']
    }
    first_games = [0]
    confidence_factors = [0.5, 0.65]
    form_lengths = [5]
    seasonal_grid_search(param_grid, first_games, confidence_factors, form_lengths)
    '''

    