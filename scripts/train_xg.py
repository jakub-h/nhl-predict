from src.xg_model_base import XGModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    model = XGModel("../data", model=LogisticRegression())
    model.fit(params=dict(penalty='l2', class_weight="balanced", max_iter=1000, C=0.01),
              normalize=True)
    y_train_pred = model.predict_xg(model.x_train)
    y_train_pred.index = model.y_train.index
    df = model.x_train
    df['xG'] = y_train_pred
    print(df)

    """
    model = XGModel("../data", model=RandomForestClassifier())
    model.grid_search(param_grid={
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [5, 20, 40, None],
        "criterion": ['gini', 'entropy']
    })

    model = XGModel("../data", model=ExtraTreesClassifier())
    model.grid_search(param_grid={
        "n_estimators": [50, 100, 200, 400],
        "criterion": ['gini', 'entropy'],
        "max_depth": [5, 10, 20, 40, 80, None],
        "min_samples_split": [2, 4, 8, 16, 32],
    })

    model = XGModel("../data", model=GradientBoostingClassifier())
    model.grid_search(param_grid={
        "loss": ['deviance', 'exponential'],
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [.0001, 0.001, 0.01, 0.1, 0.2],
        "min_samples_split": [2, 4, 8, 16, 32],
    })
    """
