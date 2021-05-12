from src.xg_model import XGModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier


if __name__ == '__main__':
    model = XGModel("../data", clf=RandomForestClassifier())
    model.fit(param_grid={
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [5, 20, 40, None],
        "criterion": ['gini', 'entropy']
    })

    model = XGModel("../data", clf=ExtraTreesClassifier())
    model.fit(param_grid={
        "n_estimators": [50, 100, 200, 400],
        "criterion": ['gini', 'entropy'],
        "max_depth": [5, 10, 20, 40, 80, None],
        "min_samples_split": [2, 4, 8, 16, 32],
    })
