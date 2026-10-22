import re
from collections.abc import Sequence

import matplotlib.pyplot as plt
import polars as pl
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


def plot_distribution_bar(
    df: pl.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str
) -> None:
    x = df[x_col].to_list()
    y = df[y_col].to_list()

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, edgecolor="k")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_boxplot(
    data: Sequence | pl.Series | Sequence[Sequence],
    labels: Sequence[str] | None = None,
    title: str = "",
    xlabel: str = "",
    figsize: tuple[int, int] = (8, 3),
    colors: Sequence[str] | None = None,
    vert: bool = False,
    show_grid: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    box = ax.boxplot(
        data,
        vert=vert,
        patch_artist=True,
        labels=labels,
        boxprops={"edgecolor": "black"},
        medianprops={"color": "red"},
    )

    if colors:
        for patch, c in zip(box["boxes"], colors, strict=False):
            patch.set_facecolor(c)
    else:
        for patch in box["boxes"]:
            patch.set_facecolor("skyblue")

    ax.set_title(title, pad=10)
    if vert:
        ax.set_ylabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    if show_grid:
        ax.grid(axis="y" if vert else "x", linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.show()


def extract_gender_from_zippia(
    url: str,
) -> tuple[float, float] | tuple[None, None]:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        container = soup.find("div", string=re.compile(r"gender ratio", re.IGNORECASE))
        if not container:
            return None, None

        parent_div = container.find_parent("div")
        if isinstance(parent_div, Tag):
            value_div = parent_div.find_all("div")[-1]

        if not value_div:
            return None, None

        text = value_div.get_text(separator="\n").strip()
        male_match = re.search(r"Male\s*[--]\s*(\d+)%", text)
        female_match = re.search(r"Female\s*[--]\s*(\d+)%", text)

        if male_match and female_match:
            male = float(male_match.group(1))
            female = float(female_match.group(1))
            return male, female

        return None, None

    except Exception as e:  # scraping: any failure means no data
        print(f"Errore su {url}: {e}")
        return None, None


def split_df_per_attribute(
    df: pl.DataFrame, attribute_name: str
) -> dict[str, pl.DataFrame]:
    dfs_per_attribute = {}
    for value in df[attribute_name].unique(maintain_order=True):
        filtered_df = df.filter(pl.col(attribute_name) == value)
        dfs_per_attribute[value] = filtered_df
    return dfs_per_attribute
