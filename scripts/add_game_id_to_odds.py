import pandas as pd


def get_id_tuple(row, stats_df):
    close_games = stats_df[
        abs(row["date"] - stats_df["date"].dt.tz_localize(None)) < "2 days"
    ]
    subdf = close_games[
        (close_games["home"] == row["home"]) & (close_games["away"] == row["away"])
    ]
    if subdf.empty:
        subdf = close_games[
            (close_games["home"] == row["away"]) & (close_games["away"] == row["home"])
        ]
    if subdf.shape[0] > 1:
        subdf["timediff"] = abs(row["date"] - subdf["date"].dt.tz_localize(None))
        subdf = subdf.sort_values("timediff").head(1)
    if subdf.shape[0] != 1:
        raise ValueError(
            f"Could not match the odds to game stats!\nsubdf\n{subdf}\nrow\n{row}"
        )
    return subdf.iloc[0, 0], subdf.iloc[0, 1]


def main():
    for season in range(2011, 2019):
        odds = pd.read_pickle(f"data/odds/{season}-{season+1}.pkl")
        stats = pd.read_pickle(f"data/games_raw/{season}-{season+1}.pickle")

        tmp_dict = {
            "season": [],
            "game_id": [],
            "date": [],
            "home": [],
            "away": [],
        }
        for game in stats:
            tmp_dict["season"].append(int(str(game["id"])[:4]))
            tmp_dict["game_id"].append(int(str(game["id"])[6:]))
            tmp_dict["date"].append(game["datetime"])
            tmp_dict["home"].append(game["teams"]["home"]["name"])
            tmp_dict["away"].append(game["teams"]["away"]["name"])

        stats_df = pd.DataFrame(tmp_dict)
        stats_df["date"] = pd.to_datetime(stats_df["date"])
        stats_df = stats_df.replace(
            {
                "Montréal Canadiens": "Montreal Canadiens",
                "Phoenix Coyotes": "Arizona Coyotes",
            }
        )
        tmp = odds.apply(lambda x: get_id_tuple(x, stats_df), axis=1)
        odds.index = pd.MultiIndex.from_tuples(tmp, names=["season", "game_id"])
        odds = odds.sort_index()
        odds.to_pickle(f"data/odds/{season}-{season+1}_gameid.pkl")


if __name__ == "__main__":
    main()
