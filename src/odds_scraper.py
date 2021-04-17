import time
from datetime import datetime
from typing import Sequence, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import selenium.common.exceptions
from bs4 import BeautifulSoup
from selenium import webdriver


class OddsScraper:
    """
    Class for scraping historical odds of NHL games.
    """
    def __init__(self,
                 base_url,
                 driver_path):
        self._base_url = base_url
        self._driver_path = driver_path

    def scrape_season(self, season: int) -> pd.DataFrame:
        """
        Scrapes and parses whole season and returns it in a DataFrame.

        :param season: int - initial year of a season (e.g. 2015 for 2015/2016 season)
        :return: pd.DataFrame - all regular season games with pregame odds (home "1", draw "X", away "2") and final score.
        """
        print(f"## OddsScraper: Scraping season {season}-{season+1} ... ", end="", flush=True)
        start = time.time()
        season_url = urljoin(self._base_url, f"nhl-{season}-{season+1}/")
        page_id = 1
        try_next_page = True
        pages = []
        while try_next_page:
            page_url = urljoin(season_url, f"results/#/page/{page_id}")
            # Open browser
            driver = webdriver.Chrome(self._driver_path)
            driver.get(season_url)
            # Change timezone
            driver.find_element_by_id("user-header-timezone-expander").click()
            timezones = driver.find_element_by_id("timezone-content")
            success = False
            while not success:
                try:
                    timezones.find_element_by_xpath("//*[text()='GMT - 4']").click()
                    success = True
                except selenium.common.exceptions.NoSuchElementException:
                    time.sleep(1)
            # Get the table from the page and parse it
            driver.get(page_url)
            table = driver.find_element_by_id("tournamentTable")
            soup = BeautifulSoup(table.text, "lxml")
            content = soup.text.split("\n")

            # Check validity (whether to continue on another page)
            if content[0] == "No data available":
                try_next_page = False
            else:
                pages.append(OddsScraper._parse_page(content))
            driver.close()
            page_id += 1
        end = time.time()
        print(f"done [{end-start} s]")
        return pd.concat(pages).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _parse_page(page_content: Sequence[str]) -> pd.DataFrame:
        """
        Goes through text content of scraped webpage, parses the results and odds, creates and returns a DataFrame.

        :param page_content: list of rows (strings)
        :return: pd.DataFrame - parsed games from this page
        """
        # Define header
        header = ["date", "home", "away", "result", "1", "X", "2"]
        games = []
        # Get season year
        season = page_content[4].split()[1]
        state = "date"
        date = None
        i_odds = None
        game = None
        for row in page_content:
            row_split = row.split()
            # Skip all Playoffs - take only regular season into account
            if "Play" in row_split and "Offs" in row_split:
                continue
            # Quit at Pre-season - take only regular season into account
            if "Pre-season" in row_split:
                break
            # Decide which state we are after non-standard row
            if state == "score/date/teams":
                if ":" in row_split[0]:
                    game.append(OddsScraper._get_result_from_score(row_split))
                    i_odds = 0
                    state = "odds"
                    continue
                elif "." in row_split[0]:
                    game.append(np.nan)
                    i_odds = 0
                    state = "odds"
                else:
                    state = "date/teams"
            # Decide whether this row is expected to be 'date' or 'teams'
            if state == "date/teams":
                if len(row_split) == 1:
                    continue
                elif row_split[-1] == "B's":
                    state = "date"
                else:
                    state = "teams"
            # Process 'date' row
            if state == "date":
                if len(row_split) > 3 and row_split[2] in season:
                    day, month, year = row_split[:3]
                    month = datetime.strptime(month, "%b").month
                    date = f"{year}-{month:02}-{day}"
                    state = "teams"
            # Process 'teams' row
            elif state == "teams":
                game = [f"{date}-{row_split[0]}"]
                if ":" not in row_split[-2] and ":" not in row_split[-1]:
                    game.extend(OddsScraper._parse_teams_row(row_split[1:], standard=False))
                    state = "score/date/teams"
                else:
                    game.extend(OddsScraper._parse_teams_row(row_split[1:], standard=True))
                    i_odds = 0
                    state = "odds"
            # Process 'score' row (only when there is some non-standard row - e.g. Winter Classic note or no result
            elif state == "score":
                game.append(OddsScraper._get_result_from_score(row_split))
                i_odds = 0
                state = "odds"
            # Process 'odds' row
            elif state == "odds":
                if i_odds < 3:
                    game.append(row_split[0])
                    i_odds += 1
                else:
                    games.append(game)
                    state = "date/teams"
        df = pd.DataFrame(games, columns=header)
        df['date'] = pd.to_datetime(df['date'])
        df.replace("-", 1.0, inplace=True)
        df['1'] = pd.to_numeric(df['1'], downcast="float")
        df['X'] = pd.to_numeric(df['X'], downcast="float")
        df['2'] = pd.to_numeric(df['2'], downcast="float")
        return df

    @staticmethod
    def _parse_teams_row(row_split: Sequence[str], standard=True) -> Tuple:
        """
        Parses row with team names.

        :param row_split: list of words in the row
        :param standard: whether the row is standard or needs some special treatment (score is missing, has some note
                         like Winter Classic etc.
        :return: tuple (home_team_name, away_team_name, result) if standard row, (home_team_name, away_team_name) otherwise.
        """
        i = row_split.index("-")
        home = row_split[:i]
        away = row_split[i + 1:]
        if not standard:
            if away[-1] in ["OT", "pen."]:
                away = away[:-1]
            return " ".join(home), " ".join(away)
        result = OddsScraper._get_result_from_score(away)
        if result == "X":
            away = away[:-2]
        else:
            away = away[:-1]
        return " ".join(home), " ".join(away), result

    @staticmethod
    def _get_result_from_score(score: Sequence[str]) -> str:
        """
        Parse score string and returns winner (or draw).
        :param score: str - in format: <home-goals>:<away_goals> [OT.|pen.]
        :return: str - "1" (home team win), "X" (draw), "2" (away team win)
        """
        if score[-1] in ['OT', 'pen.']:
            return "X"
        hg, ag = score[-1].split(":")
        result = "1" if int(hg) > int(ag) else "2"
        return result
