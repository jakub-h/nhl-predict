from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Recall, FalseNegatives, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


class XGModel:
    def __init__(self, data_path, clf, random_state=37):
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(f"'data_path' must be a string or an instance of Path class. Not {type(data_path)}.")
        self._random_state = random_state
        self._clf = clf
        self._x_train, self._x_test, self._y_train, self._y_test = self._get_datasets()

    def _get_datasets(self):
        """
        TODO
        :return:
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
        df = pd.get_dummies(df, prefix_sep="-", columns=['shot_type', 'prev_event_type'])

        return train_test_split(df.drop(columns=['outcome']), df['outcome'],
                                random_state=self._random_state, train_size=.8)

    def fit(self, param_grid, scoring="roc_auc", verbose=1, n_jobs=-1):
        """
        TODO
        :param param_grid:
        :param scoring:
        :param verbose:
        :param n_jobs:
        :return:
        """
        cv = GridSearchCV(self._clf, param_grid, scoring="balanced_accuracy", verbose=verbose, n_jobs=n_jobs)
        cv.fit(self._x_train, self._y_train)


class XGModelNN(XGModel):
    def __init__(self, data_path, model=None, random_state=37):
        super().__init__(data_path, model, random_state)

    def _normalize_train_dataset(self):
        self._min_max_scaler = MinMaxScaler()
        x_train = pd.DataFrame(self._min_max_scaler.fit_transform(self._x_train),
                               index=self._x_train.index, columns=self._x_train.columns)
        return x_train, self._y_train

    def _normalize_test_dataset(self):
        x_test = pd.DataFrame(self._min_max_scaler.transform(self._x_test),
                              index=self._x_test.index, columns=self._x_test.columns)
        return x_test, self._y_test

    def fit(self, hidden_layers="128-64-32", dropout=0.5, verbose=1, n_jobs=None):
        """
        TODO
        :param param_grid:
        :param scoring:
        :param verbose:
        :param n_jobs:
        :return:
        """
        # Get train dataset
        x_train, y_train = self._normalize_train_dataset()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8)
        print(x_train)
        print(x_train.values)

        # Calculate class weights (unbalanced dataset)
        class_weights = {0: y_train.value_counts()[1] / y_train.shape[0],
                         1: y_train.value_counts()[0] / y_train.shape[0]}

        # Define a model
        neuron_nums = hidden_layers.split("-")
        model = Sequential()
        model.add(InputLayer(input_shape=(x_train.shape[1], )))
        for layer in neuron_nums:
            model.add(Dense(int(layer), activation="relu"))
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))
        if verbose > 0:
            model.summary()

        # Compile the model
        model.compile(optimizer=Adam(),
                      loss=BinaryCrossentropy(),
                      metrics=[BinaryAccuracy(), AUC(), Recall(), FalseNegatives()])

        # Fit the model
        return model.fit(x_train, y_train, batch_size=1024, epochs=100,
                         validation_data=(x_val, y_val),
                         class_weight=class_weights,
                         use_multiprocessing=True)
