from dataset_manager_v2 import DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
import plotly.colors as cl


if __name__ == "__main__":
    dm = DatasetManager(player_names_fn='player_names_big.json')
    dm.create_base_dataset(names=False, icetime_last_n_games=40)

    y = dm._base_dataset['result']
    x = dm._base_dataset.drop(columns=['result'])
    team_enc = LabelEncoder().fit(x['home_team'])
    x['home_team'] = team_enc.transform(x['home_team'])
    x['away_team'] = team_enc.transform(x['away_team'])
    coach_enc = LabelEncoder().fit(np.concatenate([x['home_coach'], x['away_coach']]))
    x['home_coach'] = coach_enc.transform(x['home_coach'])
    x['away_coach'] = coach_enc.transform(x['away_coach'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features=None, n_jobs=3).fit(x_train, y_train)
    print(classification_report(y_train, clf.predict(x_train)))
    print(classification_report(y_test, clf.predict(x_test)))
