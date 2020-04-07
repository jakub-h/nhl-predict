import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go


if __name__ == "__main__":
    results = pd.read_csv("results/random_forest_gridsearch.csv", header=0, index_col=0) 
    print(results.loc[results['min_samples_split'] == 32].sort_values('acc_test', ascending=False).head(30))
    '''
    results.drop(columns=['dummy_teams', 'dummy_coaches', 'scale', 'max_features', 'max_depth', 'criterion'], inplace=True)
    
    y_train = results['acc_test']
    x_train = results.drop(columns=['acc_test', 'acc_train', 'ci_train', 'ci_test', 'time'])

    print(x_train)
    model = RandomForestRegressor(n_estimators = 500, n_jobs=-1).fit(x_train, y_train)
    print(model.feature_importances_)
    print(model.score(x_train, y_train))
    '''