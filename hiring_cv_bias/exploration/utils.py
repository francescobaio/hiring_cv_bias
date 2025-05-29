import re
from typing import Tuple, Union

import matplotlib.pyplot as plt
import polars as pl
import requests
from bs4 import BeautifulSoup


def localize(latitude: float) -> str:
    if latitude > 44.5:
        return "NORTH"
    if latitude < 40:
        return "SOUTH"
    return "CENTER"


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


def extract_gender_from_zippia(
    url: str,
) -> Union[Tuple[None, None], Tuple[float, float]]:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        container = soup.find("div", string=re.compile(r"gender ratio", re.I))
        if not container:
            return None, None

        parent_div = container.find_parent("div")
        value_div = parent_div.find_all("div")[-1]

        if not value_div:
            return None, None

        text = value_div.get_text(separator="\n").strip()
        male_match = re.search(r"Male\s*[–-]\s*(\d+)%", text)
        female_match = re.search(r"Female\s*[–-]\s*(\d+)%", text)

        if male_match and female_match:
            male = float(male_match.group(1))
            female = float(female_match.group(1))
            return male, female

        return None, None

    except Exception as e:
        print(f"Errore su {url}: {e}")
        return None, None
