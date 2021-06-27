from pathlib import Path

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


class XGModel:
    """
    Base xG model class.

    Model for predicting xG (expected goals) for each unblocked shot. This base class takes a Scikit-learn
    model (as a parameter in constructor) and works as a wrapper for this model.
    """
    def __init__(self, data_path, model, random_state=37, dummy=True):
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(f"'data_path' must be a string or an instance of Path class. Not {type(data_path)}.")
        self._dummy = dummy
        self._random_state = random_state
        if isinstance(model, str) or isinstance(model, Path):
            with open(model, "rb") as f:
                self._model = pickle.load(f)
        else:
            self._model = model
        self.x_train, self.x_test, self.y_train, self.y_test = self._get_datasets()
        self._scaler = StandardScaler()
        self._scaler.fit(self.x_train)

    def _get_datasets(self):
        """
        Loads datasets (train and test) for xG model. (called in the constructor)

        CSVs must be prepared on disk. See stats_scraper.convert_season_to_xg_csv().
        :return: 4-tuple - x_train, x_test, y_train, y_test (see train_test_split from sklearn).
        """
        # Get all available seasons
        csv_dir = self._data_path / "pbp_csv"
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
        if self._dummy:
            df = pd.get_dummies(df, prefix_sep="-", columns=['shot_type', 'prev_event_type'])\
                   .drop(columns=['shot_type-None'])

        return train_test_split(df.drop(columns=['outcome']), df['outcome'],
                                random_state=self._random_state, train_size=.8)

    def get_norm_train_ds(self):
        """
        Returns normalized train dataset.
        """
        norm = pd.DataFrame(self._scaler.transform(self.x_train),
                            index=self.x_train.index, columns=self.x_train.columns)
        return norm

    def get_norm_test_ds(self):
        """
        Returns normalized test dataset (based on the train dataset).
        """
        return pd.DataFrame(self._scaler.transform(self.x_test),
                            index=self.x_test.index, columns=self.x_test.columns)

    def grid_search(self, param_grid, scoring="roc_auc", verbose=1, n_jobs=-1):
        """
        Wrapper on GridSearchCV from sklearn for the underlying model.

        :param param_grid: see GridSearchCV
        :param scoring: see GridSearchCV
        :param verbose: see GridSearchCV
        :param n_jobs: see GridSearchCV
        :return: trained GridSearchCV object
        """
        cv = GridSearchCV(self._model, param_grid, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
        cv.fit(self.x_train, self.y_train)
        return cv

    def fit(self, params, normalize=True):
        """
        Fits the underlying model with given parameters.

        :param params: - dict - parameters of the model (to update)
        :param normalize: boolean - whether to normalize datasets before training
        """
        if normalize:
            x_train = self.get_norm_train_ds()
        else:
            x_train = self.x_train
        self._model.set_params(**params)
        self._model.fit(x_train, self.y_train)

    def predict(self, x):
        """
        Predicts classes for given shot-dataset.

        :param x:
        :return: y_pred - predicted classes (goal vs. no-goal)
        """
        return pd.Series(self._model.predict(x))

    def predict_xg(self, x):
        """
        Predicts xG (probabilities) for given shot-dataset.

        :param x:
        :return: y_pred - probability for each shot
        """

        xg = pd.DataFrame(self._model.predict_proba(x))
        return xg[1]

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self._model, f)
