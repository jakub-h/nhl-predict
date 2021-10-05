from multiprocessing import Pool

from nhl_predict.dataset_manager import DatasetManager


def parallel_dm_caller(season):
    print(f"> season {season}-{season+1} started ...")
    dm = DatasetManager("data")
    dm.calculate_pre_game_stats(season, last_n_games=3, save_to_csv=True)
    print(f"--> season {season}-{season+1} completed!")


if __name__ == "__main__":
    seasons = list(range(2010, 2019))
    n_jobs = 9
    print(f"# PRE_GAME stats (parallel): n_jobs={n_jobs}; seasons={seasons}")
    with Pool(n_jobs) as p:
        p.map(parallel_dm_caller, seasons)
