import time
from pathlib import Path
from typing import List

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import pandas as pd
from nhl_predict.dataset_manager import DatasetManager
from nhl_predict.game_prediction.mlp import MLP
from sklearn.preprocessing import MinMaxScaler


class Experiment:
    def __init__(
        self,
        project_root: Path,
        hidden_layers: str,
        dropout: float = 0.3,
        epochs: int = 15,
        batch_size: int = 8,
        verbose: bool = True,
    ) -> None:
        self._root_path = project_root
        self._dm = DatasetManager(project_root / "data")
        self._hidden_layers = hidden_layers
        self._dropout = dropout
        self._epochs = epochs
        self._batch_size = batch_size
        self._verbose = verbose
        self._train_times = []
        self._train_metrics = []
        self._val_metrics = []
        self.model = None
        self._scaler = None

    def run_cv(self, num_train_seasons: int = 3, num_val_seasons: int = 1):
        if self._verbose:
            print(
                f"# Experiment (CV): MLP-topology=864-{self._hidden_layers}-3; dropout="
                f"{self._dropout}; epochs={self._epochs}; batch_size={self._batch_size}"
            )
        metrics_names = None

        # Cross validation
        for x_train, x_val, y_train, y_val in self._dm.cross_validation(
            num_train_seasons, num_val_seasons, one_hot=True, normalize=True
        ):
            if self._verbose:
                print(
                    f"-> train_seasons: {x_train.index.unique(0).tolist()}; "
                    f"val_seasons: {x_val.index.unique(0).tolist()} ... ",
                    end="",
                    flush=True,
                )
            # Filter NaNs
            x_train = x_train.dropna(how="any")
            x_val = x_val.dropna(how="any")
            y_train = y_train.loc[x_train.index]
            y_val = y_val.loc[x_val.index]

            # Build model
            model = MLP(self._root_path, self._hidden_layers, self._dropout)
            model.build()

            # Train model
            start = time.time()
            model.fit(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                epochs=self._epochs,
                batch_size=self._batch_size,
                verbose=0,
            )
            end = time.time()

            self._train_times.append(end - start)
            self._train_metrics.append(
                model.evaluate(x_train, y_train, self._batch_size)
            )
            self._val_metrics.append(model.evaluate(x_val, y_val, self._batch_size))
            metrics_names = model.get_metrics_names()
            if self._verbose:
                print("done")

        df_train = pd.DataFrame(self._train_metrics, columns=metrics_names)
        df_train["time"] = self._train_times
        df_val = pd.DataFrame(self._val_metrics, columns=metrics_names)

        # Get mean and confidence intervals for each metric (using bootstrapping)
        bs_time = bs.bootstrap(df_train["time"].to_numpy(), stat_func=bs_stats.mean)
        result = {
            "train_time": f"{bs_time.value} ({bs_time.lower_bound}; {bs_time.upper_bound})"
        }
        for metric in metrics_names:
            for name, df in zip(["train", "val"], [df_train, df_val]):
                bs_res = bs.bootstrap(df[metric].to_numpy(), stat_func=bs_stats.mean)
                res_str = f"{bs_res.value} ({bs_res.lower_bound}; {bs_res.upper_bound})"
                result[f"{name}_{metric}"] = res_str

        # Add experiment params
        result["params"] = {}
        result["params"]["hidden_layers"] = self._hidden_layers
        result["params"]["dropout"] = self._dropout
        result["params"]["num_train_seasons"] = num_train_seasons
        result["params"]["num_val_seasons"] = num_val_seasons
        result["params"]["epochs"] = self._epochs
        result["params"]["batch_size"] = self._batch_size

        return result

    def train_final_model(
        self, train_seasons: List[int], val_seasons: List[int]
    ) -> pd.DataFrame:
        # Get data
        x_train, y_train = self._dm.get_dataset_by_seasons(train_seasons)
        x_val, y_val = self._dm.get_dataset_by_seasons(val_seasons)

        # Preprocess data
        self._scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
        x_train = pd.DataFrame(
            self._scaler.transform(x_train),
            index=x_train.index,
            columns=x_train.columns,
        )
        x_val = pd.DataFrame(
            self._scaler.transform(x_val), index=x_val.index, columns=x_val.columns
        )
        y_train = pd.get_dummies(y_train)
        y_val = pd.get_dummies(y_val)

        # Filter NaNs
        x_train = x_train.dropna(how="any")
        x_val = x_val.dropna(how="any")
        y_train = y_train.loc[x_train.index]
        y_val = y_val.loc[x_val.index]

        # Build model
        self.model = MLP(self._root_path, self._hidden_layers, self._dropout)
        self.model.build()

        # Train model
        history = self.model.fit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=self._epochs,
            batch_size=self._batch_size,
            verbose=1,
        )
        return history

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(
            self._scaler.transform(x),
            index=x.index,
            columns=x.columns,
        )
        # Filter NaNs
        x = x.dropna(how="any")

        return self.model.predict(x, batch_size=self._batch_size)
