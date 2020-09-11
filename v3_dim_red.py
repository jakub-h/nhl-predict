from dataset_manager_v3 import DatasetManager
from v3_main import mean_confidence_interval
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import colorlover as cl
import time
import pandas as pd
import numpy as np



def lda_seasonal_crossval(first_games, num_of_training_seasons, form_length, dummy, home_rel, minmax):
    dm = DatasetManager(base_dataset_fn=None)
    print("## Seasonal CV ##: LDA (form: {}; dummy: {}; home_rel: {}; minmax: {})".format(form_length, dummy, home_rel, minmax))
    y_true = []
    y_pred = []
    y_probs = []
    for test_season in range(2015, 2020):
        print("test season: {} (first {} games to train); train seasons: {} ---> ".format(test_season, first_games, num_of_training_seasons), end='', flush=True)
        start = time.time()
        x_train, x_test, y_train, y_test = dm.get_seasonal_split(test_season, first_games, num_of_training_seasons, form_length, dummy, home_rel, minmax)
        clf = LinearDiscriminantAnalysis(n_components=2).fit(x_train, y_train)
        end = time.time()
        y_true.append(y_test.rename('true'))
        y_probs.append(pd.DataFrame(clf.predict_proba(x_test), index=y_test.index, columns=['away', 'draw', 'home']))
        y_pred.append(pd.Series(clf.predict(x_test), index=y_test.index, name='pred'))
        print('done in {:.2f} s'.format(end - start))
    return y_true, y_pred, y_probs


if __name__ == "__main__":
    for dummy in [False, True]:
        for home_rel in [False, True]:
            for minmax in [False, True]:
                y_true, y_pred, y_probs = lda_seasonal_crossval(0, 2, 5, dummy, home_rel, minmax)
                y_true = pd.concat(y_true)
                y_pred = pd.concat(y_pred)
                y_probs = pd.concat(y_probs)
                print(10*"=", 'y_true', 10*'=')
                print(y_true.value_counts())
                print(10*"=", 'y_pred', 10*'=')
                print(y_pred.value_counts())
                print(classification_report(y_true, y_pred, zero_division=0))
                y_sure = []
                for _, row in y_probs.iterrows():
                    if sorted(row)[-1] > 0.6:
                        y_sure.append(True)
                    else:
                        y_sure.append(False)
                y_sure = pd.Series(y_sure, index=y_true.index)
                y_true_sure = y_true.loc[y_sure == True]
                y_pred_sure = y_pred.loc[y_sure == True]
                print(10*"=", 'y_true_sure', 10*'=')
                print(y_true_sure.value_counts())
                print(10*"=", 'y_pred_sure', 10*'=')
                print(y_pred_sure.value_counts())
                print(classification_report(y_true_sure, y_pred_sure, zero_division=0))

