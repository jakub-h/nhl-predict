from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import silence_tensorflow.auto  # type: ignore # noqa F401
from nhl_predict.dataset_manager import DatasetManager
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class MLP:
    """Wrapper class over a keras Sequential MLP (MultiLayer Perceptron) model."""

    def __init__(
        self,
        project_root: Path,
        hidden_layers: str = "256-64-16",
        dropout: int = 0.3,
    ) -> None:
        """Construct an instance of MLP class.

        Parameters
        ----------
        project_root : Path
            Path to the project's root dir
        hidden_layers : str, optional
            Topology of the network (hidden layers), by default "256-64-16". There is
            always input layer and softmax head defined byt the task.
        dropout : int, optional
            Dropout coefficient., by default 0.3
        """
        self._model = None
        self._project_root = project_root
        self._hidden_layers = hidden_layers
        self._dropout = dropout

    def build(self, verbose: bool = False) -> None:
        """Build and compile MLP model for prediction of NHL games.

        Parameters
        ----------
        verbose : bool, optional
            Verbosity flag, by default False
        """
        neuron_nums = self._hidden_layers.split("-")
        dm = DatasetManager(self._project_root / "data")
        x_sample, _ = dm.get_sample_data()

        # Define a model
        self._model = Sequential()
        self._model.add(InputLayer(input_shape=(x_sample.shape[1],)))
        for layer in neuron_nums:
            self._model.add(Dense(int(layer), activation="relu"))
            self._model.add(Dropout(self._dropout))
        self._model.add(Dense(3, activation="softmax", name="output"))

        # Compile the model
        self._model.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(),
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
            ],
        )

        if verbose:
            self._model.summary()

    def get_metrics_names(self) -> List:
        """
        Getter for names of metrics of the underlying Keras model.

        Returns
        -------
        list
            metrics names
        """
        return self._model.metrics_names

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_val: Optional[pd.DataFrame],
        y_val: Optional[pd.DataFrame],
        epochs: int = 15,
        batch_size: int = 8,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """Fit the model and return the training history.

        Parameters
        ----------
        x_train : pd.DataFrame
            Training input data
        y_train : pd.DataFrame
            Training true labels
        x_val : Optional[pd.DataFrame]
            Validation input data
        y_val : Optional[pd.DataFrame]
            Validation true labels
        epochs : int, optional
            Number of training epochs, by default 15
        batch_size : int, optional
            Size of the minibatch, by default 8
        verbose : int, optional
            Verbosity level, by default 0

        Returns
        -------
        history : pd.DataFrame
            Tracking of various metrics during the training.
        """
        callbacks = []
        if x_val is None or y_val is None:
            validation_data = None
        else:
            validation_data = (x_val, y_val)
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                )
            )

        history_obj = self._model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_data,
            callbacks=callbacks,
        )
        history = pd.DataFrame(
            history_obj.history,
            index=np.arange(1, len(history_obj.history["loss"]) + 1),
        )
        return history

    def evaluate(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.DataFrame, np.ndarray],
        batch_size: int = 32,
        verbose: int = 0,
    ) -> dict:
        """
        Evaluate fitted model.

        Parameters
        ----------
        x
            input array
        y_true
            target labels
        batch_size : int
            Size of the minibatch.
        verbose : int
            Verbosity level

        Returns
        -------
            Resulting metrics.
        """
        return self._model.evaluate(x, y_true, batch_size, verbose=verbose)

    def predict(
        self,
        x: pd.DataFrame,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """TODO: try out

        Parameters
        ----------
        x : pd.DataFrame
            [description]
        y_true : pd.DataFrame
            [description]
        batch_size : int, optional
            [description], by default 32
        verbose : int, optional
            [description], by default 0

        Returns
        -------
        pd.DataFrame
            [description]
        """
        result = self._model.predict(x, batch_size, verbose)
        df = pd.DataFrame(result, columns=["2", "X", "1"], index=x.index)
        return df[["1", "X", "2"]]

    @staticmethod
    def plot_training_history(history, metric):
        fig = px.line(history, y=[metric, f"val_{metric}"], markers=True)
        fig.update_layout(
            title=f"Model's {metric}", xaxis_title="epochs", yaxis_title=metric
        )
        fig.show()
