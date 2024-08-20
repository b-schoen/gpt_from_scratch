"""Note: All input dataframes are assumed to be the result of `create_and_run_eval`."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Function to create dynamic logarithmic bins and labels
def _create_log_bins(
    series: pd.Series,
    num_bins: int = 15,
) -> tuple[list[int], list[str]]:
    min_val = max(series.min(), 1)  # Ensure minimum value is at least 1 for log scale
    max_val = series.max()
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)

    # Create logarithmically spaced bins
    log_bins = np.logspace(log_min, log_max, num_bins + 1)

    # Round bin edges to integers and ensure uniqueness
    bins = sorted(set([int(round(b)) for b in log_bins]))

    # Create labels
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

    return bins, labels


def plot_heatmap(df: pd.DataFrame) -> None:

    # Rename columns for easier handling
    df = df.rename(
        columns={
            "sample.splits.max_problem_depth": "max_depth",
            "sample.splits.problem_length": "problem_length",
        }
    )

    # Ensure problem_length is integer type
    df["problem_length"] = df["problem_length"].astype(int)
    df["max_depth"] = df["max_depth"].astype(int)

    # Create dynamic logarithmic bins for problem length
    bins, labels = _create_log_bins(df["problem_length"])

    df["problem_length_bin"] = pd.cut(
        df["problem_length"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    # Group by max_depth and problem_length_bin, and calculate mean score
    grouped_df = (
        df.groupby(["max_depth", "problem_length_bin"])["score"].mean().reset_index()
    )

    # 1. Heatmap
    plt.figure(figsize=(14, 10))
    heatmap_data = grouped_df.pivot_table(
        values="score",
        index="max_depth",
        columns="problem_length_bin",
        aggfunc="mean",
    )
    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Average Accuracy"},
    )
    plt.title("Average Accuracy Heatmap: Max Problem Depth vs Problem Length Range")
    plt.tight_layout()
    plt.show()
