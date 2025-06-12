import polars as pl
import streamlit as st

INPUT_CSV = "false_positives.csv"
OUTPUT_CSV = "cv_labeled_manual.csv"

df = pl.read_csv(INPUT_CSV, separator=";")

if "index" not in st.session_state:
    st.session_state.index = 0
if "labels" not in st.session_state:
    st.session_state.labels = {}

st.title("CV Manual Labeling App")
st.markdown(
    "Classify each CV as **positive** (has a license) or **negative** (no license)."
)

if st.session_state.index < df.shape[0]:
    curr = st.session_state.index
    row = df[curr]
    candidate_id = row["CANDIDATE_ID"].item()
    cv_text = row["cv_italian"].item()

    st.subheader(f"CV #{candidate_id}")
    st.text_area("CV Content", cv_text, height=500)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Positive (has license)"):
            st.session_state.labels[candidate_id] = 1
            st.session_state.index += 1
    with col2:
        if st.button("❌ Negative (no license)"):
            st.session_state.labels[candidate_id] = 0
            st.session_state.index += 1

    st.markdown(
        f"**Progress**: {len(st.session_state.labels)} / {df.shape[0]} CVs labeled"
    )

else:
    st.success("🎉 All CVs have been labeled!")

if st.button("💾 Export Labels"):
    label_df = pl.DataFrame(
        {
            "CANDIDATE_ID": list(st.session_state.labels.keys()),
            "manual_label": list(st.session_state.labels.values()),
        }
    )
    label_df.write_csv(OUTPUT_CSV, separator=";")
    st.success(f"Labels saved to `{OUTPUT_CSV}` ✅")
