from dataset_manager import DatasetManager
import json

if __name__ == "__main__":
    dm = DatasetManager()
    dm.hr_from_csv()
    hr = dm.get_hr_table()
    print(hr.loc[hr["8477956_away"] != 0, 'away_coach'].value_counts())
    