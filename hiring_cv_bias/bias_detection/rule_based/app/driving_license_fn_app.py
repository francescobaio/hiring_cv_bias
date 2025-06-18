from collections import defaultdict
from typing import Dict, List, Tuple

import polars as pl
import streamlit as st

from hiring_cv_bias.bias_detection.rule_based.patterns import (
    driver_license_pattern_eng,
)
from hiring_cv_bias.config import CLEANED_SKILLS, DRIVING_LICENSE_FALSE_NEGATIVE_PATH


@st.cache_data(show_spinner="Loading false-negatives…")
def load_fn(path: str = DRIVING_LICENSE_FALSE_NEGATIVE_PATH) -> pl.DataFrame:
    df = pl.read_csv(path, separator=";")

    def make_snippet(txt: str) -> str:
        m = driver_license_pattern_eng.search(txt)
        if not m:
            return txt[:240] + "…"
        start = max(0, m.start() - 120)
        end = min(len(txt), m.end() + 120)
        snippet = txt[start:end]
        snippet = driver_license_pattern_eng.sub(
            lambda mo: f"<span style='color:red;font-weight:bold'>{mo.group(0)}</span>",
            snippet,
        )
        return ("…" if start else "") + snippet + ("…" if end < len(txt) else "")

    return df.with_columns(
        pl.col("cv_text")
        .map_elements(make_snippet, return_dtype=pl.Utf8)
        .alias("snippet_html"),
        pl.col("cv_text")
        .map_elements(
            lambda t: driver_license_pattern_eng.sub(
                lambda mo: f"<span style='color:red;font-weight:bold'>{mo.group(0)}</span>",
                t,
            ),
            return_dtype=pl.Utf8,
        )
        .alias("full_html"),
    )


@st.cache_data(show_spinner="Loading parser skills…")
def load_skills(path: str = CLEANED_SKILLS) -> Dict[str, List[Tuple[str, str]]]:
    df = pl.read_csv(path, separator=";").with_columns(
        pl.col("CANDIDATE_ID").cast(pl.Utf8)
    )
    skills_by_id: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for row in df.to_dicts():
        skills_by_id[row["CANDIDATE_ID"]].append((row["Skill"], row["Skill_Type"]))
    return skills_by_id


df_fn = load_fn()
skills_by_id = load_skills()


# ────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")

n = st.sidebar.slider("Number of CVs", 1, 50, 5)
shuffle = st.sidebar.checkbox("Shuffle sample", value=True)
show_full = st.sidebar.checkbox("Show full CV (not just snippet)", value=False)
search = st.sidebar.text_input("Filter by keyword / Candidate ID")

show_only_dl = st.sidebar.checkbox(
    "Show only DRIVERLICENSE skills",
    value=True,
)

# sample handling
if st.sidebar.button("Refresh"):
    st.session_state["sample"] = None
if "sample" not in st.session_state or st.session_state["sample"] is None:
    st.session_state["sample"] = df_fn.sample(n=n, shuffle=shuffle)
sample_df = st.session_state["sample"]

# apply text / ID filter
if search:
    sample_df = sample_df.filter(
        pl.any_horizontal(
            [
                pl.col("CANDIDATE_ID").cast(str).str.contains(search),
                pl.col("cv_text").str.contains(search, literal=False),
            ]
        )
    )

# ────────────────────────────────────────────────────────────────
st.title("Driving-License — False Negative Explorer")
st.markdown("<br>", unsafe_allow_html=True)

if sample_df.is_empty():
    st.info("No CVs match the current filters.")
else:
    for row in sample_df.to_dicts():
        cid = row["CANDIDATE_ID"]
        cid_str = str(cid)
        html = row["full_html"] if show_full else row["snippet_html"]

        try:
            col_cv, col_sk = st.columns([3, 2], gap="large")
        except TypeError:
            col_cv, col_sk = st.columns([3, 2])

        with col_cv:
            st.markdown(
                f"**Candidate ID:** {cid}  &nbsp;|&nbsp;  "
                f"**Skill (missed):** {row['skill']}"
            )
            st.markdown(html, unsafe_allow_html=True)

        with col_sk:
            st.markdown("**Parser skills**")
            skill_rows = skills_by_id.get(cid_str, [])

            if show_only_dl:
                skill_rows = [
                    (sk, typ) for sk, typ in skill_rows if typ == "DRIVERLICENSE"
                ]

            if not skill_rows:
                st.write("— (none)")
            else:
                for sk, _ in skill_rows:
                    st.write(sk)

        st.markdown("---")
