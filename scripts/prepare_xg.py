from src.stats_scraper import StatsScraper

if __name__ == "__main__":
    ss = StatsScraper("../data")
    for season in range(2010, 2019):
        ss.download_season(season, n_jobs=6)
        ss.convert_season_to_xg_pandas(season, save_to_csv=True)
