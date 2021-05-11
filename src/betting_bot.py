import numpy as np
import pandas as pd
from collections.abc import Callable
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


class BettingBot:
    def __init__(self, odds_path_base, bet_size):
        self._base_path = odds_path_base
        self._bet_size = bet_size

    def _get_revenue(self, game: pd.Series) -> float:
        """If the bet was won, returns bet_size * odd (revenue). Otherwise returns 0."""
        if game['win']:
            return self._bet_size * game[game['bet']]
        return 0

    def _get_deposit(self, game: pd.Series) -> int:
        """If a bet was placed, returns bet_size (deposit). Otherwise returns 0."""
        if pd.isna(game['bet']):
            return 0
        return self._bet_size

    def _bet_season(self, season: int, strategy: Callable[[pd.Series], str], **strategy_kwargs) -> pd.DataFrame:
        """
        Internal function that process bets in one season based on given betting strategy.
        Calculates revenues and deposits based on the odds.

        :param season: int - season to process
        :param strategy: function - betting strategy
        :param strategy_kwargs: dict - Additional arguments for strategy function.
        :return: pd.DataFrame - with added columns ['bet', 'win', 'revenue', 'deposit']
        """
        df = pd.read_pickle(self._base_path / f"{season}-{season + 1}.pkl")
        df['bet'] = df.apply(strategy, **strategy_kwargs, axis=1)
        df['win'] = df['result'] == df['bet']
        df['revenue'] = df.apply(self._get_revenue, axis=1)
        df['deposit'] = df.apply(self._get_deposit, axis=1)
        return df

    def bet_season(self, season: int, strategy: Callable[[pd.Series], str], verbose=1, **strategy_kwargs) -> dict:
        """
        Process bets in one season based on given betting strategy. Summaries revenues, deposits, profit and
        profit rate. If verbose > 0, it also prints the result.

        :param season: int - season to process
        :param strategy: function - betting strategy
        :param verbose: int - whether to print result or just return it
        :param strategy_kwargs: dict - Additional arguments for strategy function.
        :return: dict - {"revenue", "deposit", "profit", "profit_rate"}
        """
        df = self._bet_season(season, strategy, **strategy_kwargs)
        revenue = df['revenue'].sum()
        deposit = df['deposit'].sum()
        profit = revenue - deposit
        if deposit > 0:
            profit_rate = profit / deposit * 100
        else:
            profit_rate = np.nan
        result = {
            "revenue": revenue,
            "deposit": deposit,
            "profit": profit,
            "profit_rate": profit_rate
        }
        if verbose:
            print(f"## BettingBot (SEASON {season}/{season+1})")
            print(f"--> Strategy: {strategy.__doc__}")
            print(f"bet size:\t{self._bet_size} CZK")
            print(f"deposit:\t{deposit:.2f} CZK ({df['bet'].notna().sum()} games * {self._bet_size} CZK)")
            print(f"revenue:\t{revenue:.2f} CZK")
            print(f"profit:\t\t{profit:.2f} CZK")
            print(f"profit rate:\t{profit_rate:.2f} %")
            print()
        return result

    def bet_strategy(self, strategy: Callable[[pd.Series], str], season_range=(2005, 2018), verbose=0,
                     **strategy_kwargs) -> pd.DataFrame:
        """
        Tests given betting strategy on seasons from season_range.

        :param strategy: function - betting strategy
        :param season_range: tuple (int, int) - starting years of first and last season to use (default: (2005, 2018))
        :param verbose: int - whether to print results
        :param strategy_kwargs: dict - Additional arguments for strategy function.
        :return: pd.DataFrame - results ("revenue", "deposit", "profit", "profit_rate") from each season in season_range
        """
        results = []
        header = None
        for season in range(season_range[0], season_range[1]+1):
            season_result = self.bet_season(season, strategy, verbose, **strategy_kwargs)
            results.append(season_result.values())
            header = season_result.keys()
        return pd.DataFrame(results, columns=header, index=np.arange(season_range[0], season_range[1]+1))

    def bootstrap_strategy(self, strategy: Callable[[pd.Series], str], season_range=(2005, 2018), metric="profit_rate",
                           **strategy_kwargs) -> tuple:
        """
        Tests a strategy on given seasons and returns bootstrapped estimation of mean of given metric.

        :param strategy: function - betting strategy
        :param season_range: tuple (int, int) - starting years of first and last season to use (default: (2005, 2018))
        :param metric: - str - ('revenue', 'deposit', 'profit', 'profit_rate') default: 'profit_rate'
        :param strategy_kwargs: dict - Additional arguments for strategy function.
        :return: tuple - bootstrap result (mean (CI_low, CI_high))
        """
        df = self.bet_strategy(strategy, season_range, verbose=0, **strategy_kwargs)
        return bs.bootstrap(df[metric].to_numpy(), stat_func=bs_stats.mean)

