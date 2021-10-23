from argparse import ArgumentParser
from pathlib import Path

from nhl_predict.dataset_manager import DatasetManager
from nhl_predict.game_prediction.experiment import Experiment


def get_args():
    parser = ArgumentParser("Script for stats and histograms in age-gender-race task.")
    parser.add_argument(
        "-p", "--path", type=str, help="Path to project's root dir.", required=True
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="Plot history of the training.",
    )
    parser.set_defaults(verbose=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    project_root = Path(args.path)
    dm = DatasetManager(project_root / "data")
    train_seasons = [2011, 2012, 2013, 2014, 2015]
    val_seasons = [2016, 2017, 2018]
    exp = Experiment(
        project_root=project_root,
        hidden_layers="512-128",
        epochs=60,
        batch_size=128,
        dropout=0.1,
        verbose=True,
    )
    history = exp.train_final_model(train_seasons, val_seasons)
    if args.verbose:
        exp.model.plot_training_history(history, "auc")
        exp.model.plot_training_history(history, "loss")
    for season in train_seasons:
        x, y = dm.get_dataset_by_seasons([season])
        predictions = exp.predict(x)
        predictions.to_pickle(
            project_root / "data" / "games_predictions" / f"{season}-{season+1}.pkl"
        )
    for season in val_seasons:
        x, y = dm.get_dataset_by_seasons([season])
        predictions = exp.predict(x)
        predictions.to_pickle(
            project_root / "data" / "games_predictions" / f"{season}-{season+1}.pkl"
        )
