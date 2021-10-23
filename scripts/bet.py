import itertools
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import nhl_predict.betting.strategies as bstr
import numpy as np
import pandas as pd
import seaborn as sns
from nhl_predict.betting.bot import BettingBot


def get_args():
    parser = ArgumentParser("Script for stats and histograms in age-gender-race task.")
    parser.add_argument(
        "-p", "--path", type=str, help="Path to project's root dir.", required=True
    )
    return parser.parse_args()


def bootstrap_metric_range(path, low, high, strategy, dataset, metric):
    if dataset == "train":
        seasons = (2011, 2015)
    elif dataset == "val":
        seasons = (2016, 2018)
    else:
        raise ValueError("'dataset' must be either 'train' or 'val'")
    if low > high:
        return np.nan
    project_root = Path(path)
    bot = BettingBot(project_root, bet_size=10)
    result = bot.bootstrap_strategy(
        strategy,
        seasons,
        odd_range=(low, high),
        metric=metric,
    )
    return result.value


def bootstrap_metric_threshold(threshold, path, strategy, dataset, metric):
    if dataset == "train":
        seasons = (2011, 2015)
    elif dataset == "val":
        seasons = (2016, 2018)
    else:
        raise ValueError("'dataset' must be either 'train' or 'val'")
    bot = BettingBot(path, bet_size=10)
    result = bot.bootstrap_strategy(
        strategy,
        seasons,
        threshold=threshold,
        metric=metric,
    )
    return result.value


def search_odd_range(path, strategy, dataset, metric):
    print(f"{strategy.__name__}, {metric}, {dataset}")
    heatmap_size = 9
    low = np.linspace(1.3, 2.5, heatmap_size)
    high = np.linspace(1.5, 2.7, heatmap_size)
    ranges = itertools.product(low, high)
    args = [(path, x[0], x[1], strategy, dataset, metric) for x in ranges]

    with Pool(16) as p:
        results = p.starmap(bootstrap_metric_range, args)
    results = np.array(results).reshape((heatmap_size, heatmap_size))

    # Plot heatmap
    sns.set_theme()
    _, ax = plt.subplots(figsize=(10, 9))
    if metric == "profit":
        fmt = "n"
    else:
        fmt = ".1%"
    sns.heatmap(
        results,
        square=True,
        xticklabels=high.round(2),
        yticklabels=low.round(2),
        cmap="vlag_r",
        center=0,
        annot=True,
        fmt=fmt,
        ax=ax,
    )
    plt.title(
        f"{metric} for strategy: {strategy.__name__} ({dataset} seasons)",
        fontdict={"size": 18},
    )
    plt.xlabel("upper bound [odd]")
    plt.ylabel("lower bound [odd]")
    plt.tight_layout()
    plt.savefig(
        path
        / "outputs"
        / "plots"
        / f"heatmap_{strategy.__name__}_{metric}_{dataset}.png",
        dpi=250,
    )


def search_threshold(path, strategy, dataset, metrics):
    thresholds = np.linspace(0, 1.4, 20)
    results = {}
    for metric in metrics:
        args = [(t, path, strategy, dataset, metric) for t in thresholds]
        with Pool(16) as p:
            results[metric] = p.starmap(bootstrap_metric_threshold, args)

    results = pd.DataFrame(results, index=thresholds)

    # Plot
    with sns.axes_style("darkgrid"):
        ax = results.plot(y=["profit_rate", "bet_ratio"], legend=False)
    with sns.axes_style("white"):
        ax2 = ax.twinx()
        results.plot(y="profit", legend=False, ax=ax2, color="g")

    ax.figure.legend(bbox_to_anchor=(0.13, 0.5), loc="center left")
    ax.set_xlabel("threshold")
    ax.set_ylabel("[%]")
    ax2.set_ylabel("[CZK]")
    plt.title(f"{strategy.__name__} ({dataset} seasons)", fontdict={"size": 18})
    plt.tight_layout()
    plt.savefig(
        path / "outputs" / "plots" / f"line_{strategy.__name__}_{dataset}.png",
        dpi=300,
    )


def range_strategies(project_root):
    bot = BettingBot(project_root, 10)
    for strategy in [bstr.favorite, bstr.game_model_favorite]:
        for dataset, seasons in zip(["train", "val"], [(2011, 2015), (2016, 2018)]):
            df = bot.bet_strategy(strategy, seasons, odd_range=(1.3, 2.1))
            print(f"Strategy: {strategy.__name__} ({dataset})")
            print(df)
            print()
            df.to_latex(
                project_root
                / "outputs"
                / "tables"
                / f"{strategy.__name__}_{dataset}.tex"
            )


def threshold_strategies(project_root):
    bot = BettingBot(project_root, 10)
    for strategy, threshold in zip(
        [bstr.game_model_best_value, bstr.diff_small_underdog], [0.45, 3.5]
    ):
        for dataset, seasons in zip(["train", "val"], [(2011, 2015), (2016, 2018)]):
            df = bot.bet_strategy(strategy, seasons, threshold=threshold)
            print(f"Strategy: {strategy.__name__} ({dataset})")
            print(df)
            print()
            df.to_latex(
                project_root
                / "outputs"
                / "tables"
                / f"{strategy.__name__}_{dataset}.tex"
            )


if __name__ == "__main__":
    args = get_args()
    project_root = Path(args.path)

    range_strategies(project_root)
    threshold_strategies(project_root)
