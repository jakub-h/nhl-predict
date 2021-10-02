from src.dataset_manager import DatasetManager
from multiprocessing import Pool


def parallel_dm_caller(season):
    print(f"> season {season}-{season+1} started ...")
    dm = DatasetManager("data")
    dm.calculate_pre_game_stats(season, save_to_csv=True)
    print(f"--> season {season}-{season+1} completed!")


if __name__ == '__main__':
    seasons = list(range(2010, 2019))
    n_jobs = 9
    print(f"# PRE_GAME stats (parallel): n_jobs={n_jobs}; seasons={seasons}")
    with Pool(n_jobs) as p:
        p.map(parallel_dm_caller, seasons)
