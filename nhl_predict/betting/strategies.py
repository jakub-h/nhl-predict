import numpy as np
import pandas as pd

_COLS = ["1", "X", "2"]


def favorite(game: pd.Series, odd_range=(1, 4)) -> str:
    """Bets on favorite if the odd is in given range (default: (1, 4))."""
    odds = game["1_odd":"2_odd"]
    if odd_range[0] < odds.min() < odd_range[1]:
        return _COLS[np.argmin(odds)]
    return np.nan


def underdog(game: pd.Series, odd_range=(2, 8)) -> str:
    """Bets on underdog if the odd is in given range (default: (2, 8))."""
    odds = game[["1_odd", "2_odd"]]
    if odd_range[0] < odds.max() < odd_range[1]:
        return str(np.argmax(odds) + 1)
    return np.nan


def draw(game: pd.Series, odd_range=(2, 8)) -> str:
    """Bets on draw if the odd is in given range (default: (2, 8))."""
    if odd_range[0] < game["X_odd"] < odd_range[1]:
        return "X"
    return np.nan


def home(game: pd.Series, odd_range=(1, 8)) -> str:
    """Bets on home team if the odd is in given range (default: (1, 8))."""
    if odd_range[0] < game["1_odd"] < odd_range[1]:
        return "1"
    return np.nan


def away(game: pd.Series, odd_range=(1, 8)) -> str:
    """Bets on away if the odd is in given range (default: (1, 8))."""
    if odd_range[0] < game["2_odd"] < odd_range[1]:
        return "2"
    return np.nan


def diff_small_underdog(game: pd.Series, threshold=1.0) -> str:
    """Bets on underdog when the diff (odd_under - odd_fav) is lesser than threshold."""
    diff = game["2_odd"] - game["1_odd"]  # away - home
    if abs(diff) > threshold:
        if diff > 0:
            return "2"
        return "1"
    return np.nan


def diff_small_fav(game: pd.Series, threshold=1.0) -> str:
    """Bets on underdog when the diff (odd_under - odd_fav) is lesser than threshold."""
    diff = game["2_odd"] - game["1_odd"]  # away - home
    if abs(diff) > threshold:
        if diff > 0:
            return "1"
        return "2"
    return np.nan


def diff_small_draw(game: pd.Series, threshold=0.5) -> str:
    """Bets on draw when diff (odd_under - fav) is lesser than given threshold."""
    diff = game["2_odd"] - game["1_odd"]  # away - home
    if abs(diff) < threshold:
        return "X"
    return np.nan


def high_sum_fav(game: pd.Series, threshold=5) -> str:
    """Bets on favorite in games where sum (odd_fav + odd_under) is greater than a threshold."""
    if game["1_odd"] + game["2_odd"] > threshold:
        if game["1_odd"] > game["2_odd"]:
            return "2"
        return "1"
    return np.nan


def high_sum_underdog(game: pd.Series, threshold=5) -> str:
    """Bets on underdog in games where sum (odd_fav + odd_under) is greater than a threshold."""
    if game["1_odd"] + game["2_odd"] > threshold:
        if game["1_odd"] > game["2_odd"]:
            return "1"
        return "2"
    return np.nan


def high_sum_draw(game: pd.Series, threshold=5) -> str:
    """Bets on draw in games where sum (odd_fav + odd_under) is greater than a threshold."""
    if game["1_odd"] + game["2_odd"] > threshold:
        return "X"
    return np.nan


def game_model_favorite(game: pd.Series, odd_range=(1, 4)) -> str:
    """
    Bets on favorite of the predictive model if the odd is in given range
    """
    predictions = game["1_pred":"2_pred"]
    if any(predictions.isna()):
        return np.nan
    if odd_range[0] < predictions.min() < odd_range[1]:
        return _COLS[np.argmin(predictions)]
    return np.nan


def game_model_best_value(game: pd.Series, threshold: int = 0.2) -> str:
    """Bets on the highest positive diff between the real odd and predicted odd by
    the predictive model."""
    if any(game["1_pred":"2_pred"].isna()):
        return np.nan
    values = []
    for col in _COLS:
        values.append(game[f"{col}_odd"] - game[f"{col}_pred"])
    if np.max(values) > threshold:
        return _COLS[np.argmax(values)]
    return np.nan
