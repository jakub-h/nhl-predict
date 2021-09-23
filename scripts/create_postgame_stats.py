from src.dataset_manager import DatasetManager


if __name__ == '__main__':
    dm = DatasetManager("../data")
    for season in range(2010, 2019):
        print(f"Post-game stats of season: {season}-{season+1} ... ", end="")
        dm.calculate_post_game_stats(season, save_to_csv=True)
        print("done")
