import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLOR = ["#0C0950", "#261FB3", "#FBE4D6"]
sns.set()
TITLE_SIZE = 18
Y_TITLE = 1.01
TITLE_WEIGHT = "bold"

project_root = Path(__file__).resolve().parents[1]
FIGURE_PATH = project_root / "reports" / "figures"


def save_plot(figure_path, filename, fig):
    """
    Save the current matplotlib figure to a file.

    Args:
        figure_path (str): Directory path to save the figure.
        filename (str): Name of the file (with extension, e.g., "plot.jpg").
        fig (matplotlib.figure.Figure): The matplotlib figure to save.
    """
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    save_path = os.path.join(figure_path, filename)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.5, dpi=400)
    print(f"Figure saved to {save_path}")


def save_plotly(figure_path, filename, fig):
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    save_path = os.path.join(figure_path, filename)
    fig.write_image(save_path, width=900, scale=6)
    print(f"Figure saved to {save_path}")


def plot_multi_histogram(df, columns, savefig=False, filename="multi_histogram.jpg"):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 10))

    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df, x=col, ax=axes[i], kde=True)
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    plt.suptitle(
        "Numerical Columns Distribution",
        y=Y_TITLE,
        fontweight=TITLE_WEIGHT,
        fontsize=TITLE_SIZE,
    )
    plt.tight_layout()

    if savefig:
        save_plot(FIGURE_PATH, filename, plt.gcf())  # Pass current figure to save_plot

    plt.show()


def plot_correlation_matrix(df, columns, savefig=False, filename="heatmap.jpg"):
    fig, ax = plt.subplots(figsize=(15, 10))

    correlation = df[columns].corr()
    mask = np.triu(np.ones_like(correlation))

    sns.heatmap(
        correlation,
        cmap="Blues",
        ax=ax,
        annot=True,
        fmt=".2f",
        mask=mask,
        linewidths=0.1,
    )

    # plt.title("Correlation Matrix", fontweight='bold', loc ="left", fontsize=16)

    if savefig:
        save_plot(FIGURE_PATH, filename, plt.gcf())  # Pass current figure to save_plot

    plt.show()
