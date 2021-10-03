from sklearn.ensemble import ExtraTreesRegressor
from src.xg_model_base import XGModel

if __name__ == "__main__":
    """
    model = XGModel("../data", model=LogisticRegression())
    model.fit(params=dict(penalty='l2', class_weight="balanced", max_iter=1000, C=0.01))
    y_train_pred = model.predict_xg(model.x_train)
    y_train_pred.index = model.y_train.index
    df = model.x_train
    df['xG'] = y_train_pred
    print(df)
    model = XGModel("../data", model=RandomForestClassifier())
    model.grid_search(param_grid={
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [5, 20, 40, None],
        "criterion": ['gini', 'entropy']
    })
    """
    model = XGModel("../data", model=ExtraTreesRegressor())
    cv = model.grid_search(
        param_grid={
            "n_estimators": [20],
            "max_depth": [20],
            "min_samples_split": [128, 256, 512],
        },
        verbose=5,
    )
    print(cv.best_estimator_)
    print(cv.best_params_)
    print(cv.best_score_)
    """
    model = XGModel("../data", model=GradientBoostingClassifier())
    model.grid_search(param_grid={
        "loss": ['deviance', 'exponential'],
        "n_estimators": [50, 100, 200, 400],
        "learning_rate": [.0001, 0.001, 0.01, 0.1, 0.2],
        "min_samples_split": [2, 4, 8, 16, 32],
    })
    """
