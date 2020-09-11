import pandas as pd
import numpy as np
import json
import time
from dataset_manager_v3 import DatasetManager
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


def seasonal_crossval(params, first_games, num_of_training_seasons, form_length, dummy, home_rel, n_jobs, use_class_weights):
    dm = DatasetManager(base_dataset_fn=None)
    print("## Seasonal CV ##: RandomForest ({})".format(params))
    print("## form: {}; dummy: {}; home_rel: {}; class_weights: {}".format(form_length, dummy, home_rel, use_class_weights))
    y_true = []
    y_pred = []
    y_probs = []
    for test_season in range(2015, 2020):
        print("test season: {} (first {} games to train); train seasons: {} ---> ".format(test_season, first_games, num_of_training_seasons), end='', flush=True)
        start = time.time()
        x_train, x_test, y_train, y_test = dm.get_seasonal_split(test_season, first_games, num_of_training_seasons, form_length, dummy, home_rel, False)
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


if __name__ == "__main__":
    print(50*"=")
    params = {
        'n_estimators': 500,
        'criterion': 'entropy',
        'max_depth': None,
    }
    y_true, y_pred, y_probs = seasonal_crossval(params, 200, 2, 5, False, True, -1, False)
    pairs = zip(y_true, y_pred)

    y_true = pd.concat(y_true)
    y_pred = pd.concat(y_pred)
    y_probs = pd.concat(y_probs)
    print(10*"=", 'y_pred', 10*'=')
    print(y_pred.value_counts())
    print(classification_report(y_true, y_pred, zero_division=0, digits=4))
    y_sure = []
    for _, row in y_probs.iterrows():
        if sorted(row)[-1] > 0.45:
            y_sure.append(True)
        else:
            y_sure.append(False)
    y_sure = pd.Series(y_sure, index=y_true.index)
    y_true_sure = y_true.loc[y_sure == True]
    y_pred_sure = y_pred.loc[y_sure == True]
    print(10*"=", 'y_pred_sure', 10*'=')
    print(y_pred_sure.value_counts())
    print(classification_report(y_true_sure, y_pred_sure, zero_division=0, digits=4))