
# data.py
"""
Plotting utilities for the Heart Disease ML Project.

- plot_target_distribution(): Bar plot of class counts for the classification target.
- plot_numeric_summary(): Correlation heatmap or boxplot summary for selected numeric features.

These helpers load data from data1.load_dataset() by default so they can be used standalone.
Plots are saved under models/plots/ by default.
"""

from pathlib import Path
from typing import Iterable, Optional, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source the dataset from data1.py
from data import load_dataset  # re-exported convenience


PLOTS_DIR = Path("models") / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if df is not None else load_dataset()


def plot_target_distribution(
    df: Optional[pd.DataFrame] = None,
    target_col: str = "presence",
    save: bool = True,
    show: bool = False,
    filename: Optional[str] = None,
):
    """
    Plot a bar chart of class counts for the target.

    Parameters
    ----------
    df : DataFrame, optional
        Dataset. If None, loads via load_dataset().
    target_col : str
        Name of the classification target column.
    save : bool
        If True, saves the figure under models/plots/.
    show : bool
        If True, displays the figure (useful in notebooks).
    filename : str, optional
        Custom filename for the saved plot.
    """
    data = _resolve_df(df)
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    counts = data[target_col].value_counts(dropna=False).sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Target Distribution – {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = None
    if save:
        out_name = filename or f"target_distribution_{target_col}.png"
        out_path = PLOTS_DIR / out_name
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_numeric_summary(
    df: Optional[pd.DataFrame] = None,
    numeric_cols: Optional[Iterable[str]] = None,
    mode: Literal["heatmap", "boxplot"] = "heatmap",
    save: bool = True,
    show: bool = False,
    filename: Optional[str] = None,
):
    """
    Plot correlation heatmap or boxplot summary for key numeric features.

    Parameters
    ----------
    df : DataFrame, optional
        Dataset. If None, loads via load_dataset().
    numeric_cols : iterable of str, optional
        Columns to include. If None, auto-select numeric dtype columns.
    mode : {'heatmap', 'boxplot'}
        Plot type. 'heatmap' draws Pearson correlation heatmap.
        'boxplot' draws per-feature boxplots.
    save : bool
        If True, saves the figure under models/plots/.
    show : bool
        If True, displays the figure (useful in notebooks).
    filename : str, optional
        Custom filename for the saved plot.
    """
    data = _resolve_df(df)
    if numeric_cols is None:
        num_df = data.select_dtypes(include=[np.number])
    else:
        missing = [c for c in numeric_cols if c not in data.columns]
        if missing:
            raise KeyError(f"Columns not in dataframe: {missing}")
        num_df = data[list(numeric_cols)]

    if num_df.shape[1] == 0:
        raise ValueError("No numeric columns available to plot.")

    fig, ax = plt.subplots(figsize=(8, 6))

    if mode == "heatmap":
        corr = num_df.corr(numeric_only=True)
        im = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
        ax.set_title("Correlation Heatmap (Pearson)")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    elif mode == "boxplot":
        ax.boxplot(num_df.values, labels=num_df.columns, vert=True, patch_artist=False)
        ax.set_title("Numeric Feature Summary (Boxplots)")
        ax.set_ylabel("Value")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        raise ValueError("mode must be 'heatmap' or 'boxplot'")

    out_path = None
    if save:
        default_name = f"numeric_{mode}.png"
        out_name = filename or default_name
        out_path = PLOTS_DIR / out_name
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Generate default plots when run as a script.
    # Plot 1: Target distribution for the classification target 'presence'
    out1 = plot_target_distribution()

    # Plot 2: Correlation heatmap for numeric columns
    out2 = plot_numeric_summary(mode="heatmap")

    print("Saved plots:")
    if out1:
        print(f"- {out1}")
    if out2:
        print(f"- {out2}")
