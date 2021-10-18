import itertools
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import nhl_predict.betting.strategies as bstr
import numpy as np
import seaborn as sns
from nhl_predict.betting.bot import BettingBot


def get_args():
    parser = ArgumentParser("Script for stats and histograms in age-gender-race task.")
    parser.add_argument(
        "-p", "--path", type=str, help="Path to project's root dir.", required=True
    )
    return parser.parse_args()


def profit_bootstrap(path, low, high):
    print(low, high)
    if low > high:
        return np.nan
    project_root = Path(path)
    bot = BettingBot(project_root, bet_size=10)
    profit = bot.bootstrap_strategy(
        bstr.game_model_favorite,
        (2011, 2015),
        odd_range=(low, high),
        metric="profit",
    )
    return profit.value


def search_odd_range(path):
    heatmap_size = 8
    low = np.linspace(1.3, 2.0, heatmap_size)
    high = np.linspace(1.8, 2.5, heatmap_size)
    ranges = itertools.product(low, high)
    args = [(path, x[0], x[1]) for x in ranges]

    with Pool(16) as p:
        profits = p.starmap(profit_bootstrap, args)
    profits = np.array(profits).reshape((heatmap_size, heatmap_size))

    # Plot heatmap
    sns.set_theme()
    _, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(
        profits,
        square=True,
        xticklabels=high.round(2),
        yticklabels=low.round(2),
        cmap="vlag_r",
        center=0,
        annot=True,
        fmt=".1f",
        ax=ax,
    )
    plt.title(
        "Profit for strategy: model_favorite (train seasons)",
        fontdict={"size": 18},
    )
    plt.xlabel("upper bound [odd for favorite]")
    plt.ylabel("lower bound [odd for favorite]")
    plt.tight_layout()
    plt.show()


def bet_strategy(bot):
    print(
        bot.bet_strategy(
            bstr.game_model_favorite, season_range=(2011, 2015), odd_range=(1.5, 3)
        )
    )
    print(
        bot.bet_strategy(
            bstr.game_model_favorite, season_range=(2016, 2018), odd_range=(1.5, 3)
        )
    )


def main():
    args = get_args()
    project_root = Path(args.path)
    search_odd_range(project_root)


if __name__ == "__main__":
    main()
