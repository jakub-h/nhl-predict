from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV


class XGModel:
    def __init__(self, data_path, clf, random_state=42):
        if isinstance(data_path, Path):
            self._data_path = data_path
        elif isinstance(data_path, str):
            self._data_path = Path(data_path)
        else:
            raise ValueError(f"'data_path' must be a string or an instance of Path class. Not {type(data_path)}.")
        self._random_state = random_state
        self._clf = clf

    def _get_whole_dataset(self):
        """
        TODO documentation
        :return:
        """
        # Get all available seasons
        csv_dir = self._data_path / "pbp_csv"
        csvs = []
        for filename in sorted(csv_dir.iterdir()):
            if filename.suffix == ".csv":
                csvs.append(pd.read_csv(filename, index_col=0))
        df = pd.concat(csvs)

        # Clean data
        df['shot_type'].fillna("None", inplace=True)
        df['shot_type'] = df['shot_type'].apply(lambda x: x.split(" ")[0])
        df.fillna(0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create dummy variables from categorical
        df = pd.get_dummies(df, prefix_sep="-", columns=['shot_type', 'prev_event_type'])
        return df

    def fit(self, param_grid, verbose=1, n_jobs=-1):
        df = self._get_whole_dataset()
        x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['outcome']), df['outcome'],
                                                            random_state=self._random_state,
                                                            train_size=.8)
        cv = GridSearchCV(self._clf, param_grid, scoring="balanced_accuracy", verbose=verbose, n_jobs=n_jobs)
        cv.fit(x_train, y_train)
        with open(self._data_path / ".." / "outputs" / "gridsearch.txt", "a") as f:
            f.write(f"{self._clf}\n")
            f.write(f"{cv.best_estimator_}\n")
            f.write(f"{cv.best_params_}\n")
            f.write(f"{cv.best_score_}\n")
            f.write(30 * "-")
            f.write("\n")
