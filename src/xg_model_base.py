from pathlib import Path

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class XGModel:
    """
    Base xG model class.

    Model for predicting xG (expected goals) for each unblocked shot. This base class takes a Scikit-learn
    model (as a parameter in constructor) and works as a wrapper for this model.
    """
    def __init__(self, data_path, model, random_state=37):
        self._random_state = random_state

        # Data path
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(f"'data_path' must be a string or an instance of Path class. Not {type(data_path)}.")

        # Load model if already fitted and stored
        if isinstance(model, str) or isinstance(model, Path):
            with open(model, "rb") as f:
                self._model = pickle.load(f)
        # Initiate fresh unfitted model
        else:
            self._model = model

        # Load datasets and split into train/test
        self.x_train, self.x_test, self.y_train, self.y_test = self._get_datasets()

        # Fit data scaler
        self._scaler = StandardScaler()
        self._scaler.fit(self.x_train.drop(columns=['game_id']))

        # Init output scaler (Optimized for ExtraTreeRegressor!)
        self._output_scaler = Pipeline([('quantile', QuantileTransformer(output_distribution="normal")),
                                        ('minmax', MinMaxScaler())])

    def _get_datasets(self):
        """
        Loads datasets (train and test) for xG model. (called in the constructor)

        CSVs must be prepared on disk. See stats_scraper.convert_season_to_xg_csv().
        :return: 4-tuple - x_train, x_test, y_train, y_test (see train_test_split from sklearn).
        """
        # Get all available seasons
        csv_dir = self._data_path / "xg_pbp"
        csvs = []
        for filename in sorted(csv_dir.iterdir()):
            if filename.suffix == ".csv":
                csvs.append(pd.read_csv(filename, index_col=0))
        df = pd.concat(csvs)

        # Clean data
        df['shot_type'].fillna("None", inplace=True)
        df['shot_type'] = df['shot_type'].apply(lambda x: x.split(" ")[0])
        df.fillna(0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create dummy variables from categorical
        df = pd.get_dummies(df, prefix_sep="-", columns=['shot_type', 'prev_event_type'])\
               .drop(columns=['shot_type-None'])

        return train_test_split(df.drop(columns=['outcome']), df['outcome'],
                                random_state=self._random_state, train_size=.8)

    def scale_dataset(self, x):
        """
        Returns scaled (StandardScaler) x.

        Also drops game_id columns.
        """
        scaled = pd.DataFrame(self._scaler.transform(x.drop(columns=['game_id'])),
                              index=x.index, columns=x.columns.drop("game_id"))
        return scaled

    def grid_search(self, param_grid, scoring="roc_auc", verbose=1, n_jobs=-1):
        """
        Wrapper on GridSearchCV from sklearn for the underlying model.

        :param param_grid: see GridSearchCV
        :param scoring: see GridSearchCV
        :param verbose: see GridSearchCV
        :param n_jobs: see GridSearchCV
        :return: trained GridSearchCV object
        """
        x_train = self.scale_dataset(self.x_train)
        cv = GridSearchCV(self._model, param_grid, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
        cv.fit(x_train, self.y_train)
        return cv

    def fit(self, params):
        """
        Fits the underlying model with given parameters.

        :param params: - dict - parameters of the model (to update)
        """
        # Prepare dataset and model
        x_train = self.scale_dataset(self.x_train)
        self._model.set_params(**params)
        # Fit the model
        self._model.fit(x_train, self.y_train)
        # Predict y_train and fit the scaler based on it
        y_train_xg = self._model.predict(x_train).reshape(-1, 1)
        self._output_scaler.fit(y_train_xg)

    def predict(self, x):
        """
        Predicts classes for given shot-dataset.

        :param x:
        :return: y_pred - predicted classes (goal vs. no-goal)
        """
        x_scaled = self.scale_dataset(x)
        y = self._model.predict(x_scaled).reshape(-1, 1)
        return pd.DataFrame(self._output_scaler.transform(y), columns=['xG'])['xG']

    '''
    def predict_xg(self, x):
        """
        Predicts xG (probabilities) for given shot-dataset.

        Necessary for some sklearn models (e.g. LogisticRegression)

        :param x:
        :return: y_pred - probability for each shot
        """
        x_scaled = self.scale_dataset(x)
        xg = pd.DataFrame(self._model.predict_proba(x_scaled))
        return xg[1]
    '''

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self._model, f)
