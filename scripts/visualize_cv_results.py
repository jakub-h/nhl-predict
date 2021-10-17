import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_df():
    dir_path = Path("results")
    results = []
    for filename in sorted(dir_path.iterdir()):
        if filename.suffix == ".json":
            with open(filename, "r") as f:
                experiment = json.load(f)
                tmp_dict = {}
                for key in experiment["params"]:
                    tmp_dict[key] = experiment["params"][key]
                for key in list(experiment.keys())[1:]:
                    tmp_dict[key] = float(experiment[key].split()[0])
                results.append(tmp_dict)
    df = pd.DataFrame(results).drop(columns=["num_train_seasons", "num_val_seasons"])
    return df


def plot_boxplot_one_col(df: pd.DataFrame, xaxis, yaxis):
    df = df[[yaxis, xaxis]]
    sns.set_theme()
    sns.set(rc={"figure.figsize": (12, 6)})

    ax = sns.boxplot(data=df, x=xaxis, y=yaxis)
    ax.set(xlabel=xaxis, ylabel=yaxis.split("_")[1])
    plt.savefig(f"outputs/plots/boxplot_{xaxis}_{yaxis}.png")
    plt.tight_layout()
    plt.clf()


def plot_boxplot_two_cols(df: pd.DataFrame, xaxis, yaxis: list):
    df = df[yaxis + [xaxis]].melt(id_vars=xaxis)
    sns.set_theme()
    sns.set(rc={"figure.figsize": (12, 6)})

    ax = sns.boxplot(data=df, x=xaxis, y="value", hue="variable")
    ax.set(xlabel=xaxis, ylabel=yaxis[0].split("_")[1])
    plt.savefig(f"outputs/plots/boxplot_{xaxis}_{yaxis[0].split('_')[1]}.png")
    plt.clf()


def main():
    df = get_df()
    df = df[(df["val_auc"] > 0.614)]
    print(df.sort_values("val_auc", ascending=False))
    for param in ["batch_size", "hidden_layers", "dropout"]:
        plot_boxplot_one_col(df, param, "val_auc")
        plot_boxplot_two_cols(df, param, ["val_auc", "train_auc"])


if __name__ == "__main__":
    main()
