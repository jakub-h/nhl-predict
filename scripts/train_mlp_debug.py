from pathlib import Path

import pandas as pd
from nhl_predict.dataset_manager import DatasetManager
from nhl_predict.game_prediction.mlp import MLP

if __name__ == "__main__":
    dm = DatasetManager("data")
    model = MLP(Path("/home/jhruska/projects/nhl-predict"))
    x_train, y_train = dm.get_sample_data()
    y_train = pd.get_dummies(y_train)
    print(x_train)
    print(y_train)
    model.build(verbose=True)
    history = model.fit(x_train, y_train, x_val=None, y_val=None, verbose=1)
    model.plot_training_history(history, "accuracy")
