from dataset_manager import DatasetManager
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display


if __name__ == "__main__":
    dm = DatasetManager(games_hr_fn=None)
    for dt in [False, True]:
        for dc in [False, True]:
            for sc in [None, 'std', 'minmax']:
                train_scores = []
                test_scores = []
                for i in range(5):
                    x_train, x_test, y_train, y_test = dm.get_fold(i, dummy_teams=dt, dummy_coaches=dc, scale=sc)
                    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
                    clf.fit(x_train, y_train)
                    train_scores.append(clf.score(x_train, y_train))
                    test_scores.append(clf.score(x_test, y_test))
                print("dt: {}, dc: {}, scale: {} --> train: {:.4f}".format(dt, dc, sc, np.mean(train_scores)))
                print("dt: {}, dc: {}, scale: {} --> test: {:.4f}".format(dt, dc, sc, np.mean(test_scores)))
    '''
    feature_importances = pd.Series(clf.feature_importances_)
    feature_names = []
    for feature in x_train.columns[feature_importances.sort_values().keys()]:
        if (feature.endswith("_home") or feature.endswith("_away")) and not feature.startswith("other"):
            player_id, h_a = feature.split("_")
            feature_names.append("{} - {}".format(dm._player_names['id_to_name'][player_id], h_a))
        else:
            feature_names.append(feature)    

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=feature_names,
            x=feature_importances.sort_values().values,
            orientation='h'
        )
    )
    fig.show()
    '''
    