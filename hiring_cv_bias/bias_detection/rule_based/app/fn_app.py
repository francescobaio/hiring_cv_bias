import re
from collections import defaultdict
from typing import Callable, Dict, List, Pattern, Tuple, TypedDict

import polars as pl
import streamlit as st

from hiring_cv_bias.bias_detection.rule_based.extractors import LANGUAGE_REGEXES_EN
from hiring_cv_bias.bias_detection.rule_based.patterns import (
    driver_license_pattern_eng,
    jobs_pattern,
    languages_pattern_eng,
)
from hiring_cv_bias.config import (
    CLEANED_SKILLS,
    DRIVING_LICENSE_FALSE_NEGATIVES_PATH,
    JOB_TITLE_FALSE_NEGATIVES_PATH,
    LANGUAGE_SKILL_FALSE_NEGATIVES_PATH,
)
from hiring_cv_bias.utils import load_data


class CategoryConf(TypedDict):
    fn_path: str
    regex: Pattern[str]
    tag_pred: Callable[[str], bool]
    title: str


CATEGORIES: Dict[str, CategoryConf] = {
    "Driver License": {
        "fn_path": DRIVING_LICENSE_FALSE_NEGATIVES_PATH,
        "regex": driver_license_pattern_eng,
        "tag_pred": lambda t: t == "DRIVERLICENSE",
        "title": "Driving-License — False Negative Explorer",
    },
    "Language Skill": {
        "fn_path": LANGUAGE_SKILL_FALSE_NEGATIVES_PATH,
        "regex": languages_pattern_eng,
        "tag_pred": lambda t: t == "Language_Skill",
        "title": "Language-Skill — False Negative Explorer",
    },
    "Job Title": {
        "fn_path": JOB_TITLE_FALSE_NEGATIVES_PATH,
        "regex": jobs_pattern,
        "tag_pred": lambda t: t == "Job_title",
        "title": "Job-title — False Negative Explorer",
    },
}

choice = st.sidebar.selectbox("Select error category", list(CATEGORIES.keys()))
CFG: CategoryConf = CATEGORIES[choice]

if st.session_state.get("current_cat") != choice:
    st.session_state.pop("sample", None)
    st.session_state["current_cat"] = choice


# ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading raw false-negatives…")
def load_fn_raw(path: str) -> pl.DataFrame:
    return pl.read_csv(
        path, separator=";", columns=["CANDIDATE_ID", "cv_text", "skill"]
    ).with_columns(pl.col("CANDIDATE_ID").cast(pl.Utf8))


@st.cache_data(show_spinner="Loading parser skills…")
def load_skills(path: str = CLEANED_SKILLS) -> Dict[str, List[Tuple[str, str]]]:
    df = load_data(path).with_columns(pl.col("CANDIDATE_ID").cast(pl.Utf8))
    d: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in df.to_dicts():
        d[r["CANDIDATE_ID"]].append((r["Skill"], r["Skill_Type"]))
    return d


def make_snippet(text: str, rx: Pattern[str]) -> str:
    m = rx.search(text)
    if not m:
        return text[:240] + "…"
    start, end = max(0, m.start() - 120), min(len(text), m.end() + 120)
    snippet = text[start:end]
    snippet = rx.sub(
        lambda mo: f"<span style='color:red;font-weight:bold'>{mo.group(0)}</span>",
        snippet,
    )
    return ("…" if start else "") + snippet + ("…" if end < len(text) else "")


# ────────────────────────────────────────────────────────────────
df_fn_raw = load_fn_raw(CFG["fn_path"])
skills_full = load_skills()

# ────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
n = st.sidebar.slider("Number of CVs", 1, 50, 5)
show_full = st.sidebar.checkbox("Show full CV (not just snippet)", False)
search = st.sidebar.text_input("Filter by keyword / Candidate ID")
show_only = st.sidebar.checkbox("Show only matching skills", True)

if st.sidebar.button("Refresh"):
    st.session_state.pop("sample", None)

if "sample" not in st.session_state:
    st.session_state["sample"] = df_fn_raw.sample(n=n, shuffle=True)
sample_df = st.session_state["sample"]

# ────────────────────────────────────────────────────────────────
if search:
    if search.isdigit():
        sample_df = df_fn_raw.filter(pl.col("CANDIDATE_ID") == search)
    else:
        sample_df = df_fn_raw.filter(
            pl.any_horizontal(
                [
                    pl.col("CANDIDATE_ID").str.contains(search, literal=True),
                    pl.col("cv_text").str.contains(search, literal=False),
                ]
            )
        )


# ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating snippets…")
def annotate(text: str, rx: Pattern[str]) -> str:
    highlighted_text = rx.sub(
        lambda mo: f"<span style='color:red;font-weight:bold'>{mo.group(0)}</span>",
        text,
    )
    return highlighted_text


ids = set(sample_df["CANDIDATE_ID"].to_list())
skills_by_id = {cid: skills_full.get(cid, []) for cid in ids}

# ────────────────────────────────────────────────────────────────
st.title(CFG["title"])
st.markdown("<br>", unsafe_allow_html=True)

if sample_df.is_empty():
    st.info("No CVs match the current filters.")
else:
    for row in sample_df.to_dicts():
        cid = row["CANDIDATE_ID"]

        if choice == "Driver License":
            regex = CFG["regex"]
        elif choice == "Language Skill":
            regex = LANGUAGE_REGEXES_EN[row["skill"]]
        elif choice == "Job Title":
            regex = re.compile(rf"\b{row['skill']}\b")

        html = (
            annotate(row["cv_text"], regex)
            if show_full
            else annotate(make_snippet(row["cv_text"], regex), regex)
        )

        col_cv, col_sk = st.columns([3, 2], gap="large")
        with col_cv:
            st.markdown(
                f"**Candidate ID:** {cid}  |  **Skill (missed):** {row['skill']}"
            )
            st.markdown(html, unsafe_allow_html=True)

        with col_sk:
            st.markdown("**Parser skills**")
            rows = skills_by_id.get(cid, [])
            if show_only:
                rows = [(sk, t) for sk, t in rows if CFG["tag_pred"](t)]
            if not rows:
                st.write("— (none)")
            else:
                for sk, _ in rows:
                    st.write(sk)

        st.markdown("---")
