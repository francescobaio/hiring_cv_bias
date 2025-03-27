import polars as pl
from typing import List, Dict
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re


def load_data(filepath):
    return pl.read_csv(filepath, separator=";")


def load_excel_sheets(path: str, sheets: List[str]) -> Dict[str, pl.DataFrame]:
    return {sheet: pl.read_excel(path, sheet_name=sheet) for sheet in sheets}


def plot_distribution_bar(
    df: pl.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str
):
    x = df[x_col].to_list()
    y = df[y_col].to_list()

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def extract_gender_from_zippia(url):
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
