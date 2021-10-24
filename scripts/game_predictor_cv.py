import datetime
import itertools
import json
from argparse import ArgumentParser
from pathlib import Path

from nhl_predict.game_prediction.experiment import Experiment
from tqdm import tqdm


def get_args():
    parser = ArgumentParser("Script for game-predictor cv gridsearch.")
    parser.add_argument(
        "-p", "--path", type=str, help="Path to project's root dir.", required=True
    )

    return parser.parse_args()


def main():
    args = get_args()
    project_root = Path(args.path)
    param_grid = [
        [
            "64",
            "64-16",
            "64-16-8",
            "128",
            "128-32",
            "128-32-8",
            "256",
            "256-64",
            "256-64-16",
            "512",
            "512-128",
            "512-128-32",
        ],  # topology
        [60],  # epochs
        [64, 128, 256, 512],  # batch_size
        [0.05, 0.1, 0.2],  # dropout
    ]
    param_list = list(itertools.product(*param_grid))

    for topology, epochs, batch_size, dropout in tqdm(param_list):
        timestamp = datetime.datetime.now()
        exp = Experiment(
            project_root=project_root,
            hidden_layers=topology,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
            verbose=False,
        )
        result = exp.run_cv(num_train_seasons=3, num_val_seasons=1)
        filename = (
            f"{timestamp.date()}_{timestamp.hour:02d}{timestamp.minute:02d}"
            f"{timestamp.second:02d}"
        )
        with open(project_root / f"results/{filename}.json", "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
