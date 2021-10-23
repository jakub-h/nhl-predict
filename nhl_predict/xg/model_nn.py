import pandas as pd
from nhl_predict.xg.model_base import XGModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import AUC, RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class XGModelNN(XGModel):
    """
    Neural network version of a xG model.
    """

    def __init__(
        self, data_path, hidden_layers="64-32-16", dropout=0.3, random_state=37
    ):
        super().__init__(data_path, None, random_state)
        self._topology = hidden_layers
        self._dropout = dropout

    def fit(self, verbose=2, n_epochs=100, batch_size=1024, **kwargs):
        """
        Fits a feedforward neural network for predicting xG.

        :param verbose: verbosity level (default: 1)
        :param n_epochs: number of epochs in training
        :param batch_size: size of the mini-batch
        :return: history of the training
        """
        # Normalize
        x_train = self.get_norm_train_ds()

        # Split train dataset ot train and val
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, self.y_train, train_size=0.8
        )

        # Calculate class weights (unbalanced dataset)
        class_weights = {
            0: y_train.value_counts()[1] / y_train.shape[0],
            1: y_train.value_counts()[0] / y_train.shape[0],
        }

        # Define a model
        neuron_nums = self._topology.split("-")
        self._model = Sequential()
        self._model.add(InputLayer(input_shape=(x_train.shape[1],)))
        for layer in neuron_nums:
            self._model.add(Dense(int(layer), activation="relu"))
            self._model.add(Dropout(self._dropout))
        self._model.add(Dense(1, activation="sigmoid"))
        if verbose > 0:
            self._model.summary()

        # Compile the model
        self._model.compile(
            optimizer=Adam(),
            loss=MeanSquaredError(),
            metrics=[AUC(), RootMeanSquaredError()],
        )

        # Fit the model and return history
        return self._model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_val, y_val),
            class_weight=class_weights,
        )

    def predict_xg(self, x, batch_size=1024):
        xg = self._model.predict(self._scaler.transform(x), batch_size=batch_size)
        return pd.Series(
            xg.reshape(
                (xg.shape[0]),
            )
        )
