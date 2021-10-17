from argparse import ArgumentParser
from pathlib import Path

from nhl_predict.dataset_manager import DatasetManager
from nhl_predict.game_prediction.experiment import Experiment


def get_args():
    parser = ArgumentParser("Script for stats and histograms in age-gender-race task.")
    parser.add_argument(
        "-p", "--path", type=str, help="Path to project's root dir.", required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    project_root = Path(args.path)
    dm = DatasetManager(project_root / "data")
    train_seasons = [2011, 2012, 2013, 2014, 2015, 2016]
    val_seasons = [2017, 2018]
    exp = Experiment(
        project_root=project_root,
        hidden_layers="512-128",
        epochs=60,
        batch_size=128,
        dropout=0.1,
        verbose=True,
    )
    history = exp.train_final_model(train_seasons, val_seasons)
    x_val, y_val = dm.get_dataset_by_seasons(val_seasons)
    x_train, y_train = dm.get_dataset_by_seasons(train_seasons)
    train_pred = print(exp.predict(x_train))
