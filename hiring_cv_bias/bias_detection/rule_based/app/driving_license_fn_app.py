import polars as pl
import streamlit as st

from hiring_cv_bias.bias_detection.rule_based.patterns import driver_license_pattern_eng


def load_fn(path: str = "false_negatives.csv") -> pl.DataFrame:
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

    df = df.with_columns(
        pl.col("cv_text")
        .map_elements(make_snippet, return_dtype=pl.Utf8)
        .alias("snippet_html"),
        pl.col("cv_text")
        .map_elements(  # full text highlighted
            lambda t: driver_license_pattern_eng.sub(
                lambda mo: f"<span style='color:red;font-weight:bold'>{mo.group(0)}</span>",
                t,
            ),
            return_dtype=pl.Utf8,
        )
        .alias("full_html"),
    )
    return df


df_fn = load_fn()

# -----------------------------------------------------------------------------
st.sidebar.header("Controls")

n = st.sidebar.slider("Number of CVs", 1, 50, 5)
shuffle = st.sidebar.checkbox("Shuffle sample", value=True)
show_full = st.sidebar.checkbox("Show full CV (not just snippet)", value=False)
search = st.sidebar.text_input("Filter by keyword / Candidate ID")

# Handle refresh button & initial sample
if st.sidebar.button("Refresh"):
    st.session_state["sample"] = None
if "sample" not in st.session_state or st.session_state["sample"] is None:
    st.session_state["sample"] = df_fn.sample(n=n, shuffle=shuffle)
sample_df = st.session_state["sample"]


if search:
    sample_df = sample_df.filter(
        pl.any_horizontal(
            [
                pl.col("CANDIDATE_ID").cast(str).str.contains(search),
                pl.col("cv_text").str.contains(search, literal=False),
            ]
        )
    )

# -----------------------------------------------------------------------------
st.title("Driving License — False Negative Explorer")

if sample_df.is_empty():
    st.info("No CVs match the current filters.")
else:
    for row in sample_df.to_dicts():
        st.markdown(
            f"**Candidate ID:** {row['CANDIDATE_ID']}  &nbsp;|&nbsp;  "
            f"**Skill (missed):** {row['skill']}"
        )
        html = row["full_html"] if show_full else row["snippet_html"]
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("---")
