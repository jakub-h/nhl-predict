from argparse import ArgumentParser
from pathlib import Path

import nhl_predict.betting.strategies as bstr
from nhl_predict.betting.bot import BettingBot


def get_args():
    parser = ArgumentParser("Script for stats and histograms in age-gender-race task.")
    parser.add_argument(
        "-p", "--path", type=str, help="Path to project's root dir.", required=True
    )
    return parser.parse_args()


def main():
    args = get_args()
    project_root = Path(args.path)
    bot = BettingBot(project_root, bet_size=10)
    print(
        bot.bet_strategy(
            bstr.game_model_basic, season_range=(2011, 2015), threshold=0.5
        )
    )
    print(
        bot.bet_strategy(
            bstr.game_model_basic, season_range=(2016, 2018), threshold=0.5
        )
    )
    """
    for threshold in np.linspace(0.2, 0.6, 20):
        res = bot.bootstrap_strategy(
            bstr.game_model_basic, (2011, 2015), threshold=threshold
        )
        print(f"{threshold:.2f}: {res}")
    """


if __name__ == "__main__":
    main()
