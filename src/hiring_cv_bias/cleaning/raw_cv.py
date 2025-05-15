import random
import re
from typing import List, Pattern, Tuple

import matplotlib.pyplot as plt
import polars as pl
from langdetect import DetectorFactory, detect


def add_length_column(
    df: pl.DataFrame, text_col: str = "CV_text_anon", length_col: str = "len_anon"
) -> pl.DataFrame:
    """ """
    return df.with_columns(
        [
            pl.col(text_col)
            .map_elements(lambda s: len(s.strip()) if s else 0, return_dtype=pl.Int64)
            .alias(length_col)
        ]
    )


def find_and_print_short_cvs(
    df: pl.DataFrame,
    length_col: str = "len_anon",
    threshold: int = 300,
    id_col: str = "CANDIDATE_ID",
    text_col: str = "CV_text_anon",
) -> pl.DataFrame:
    """ """
    # 1) Select short CVs
    too_short = df.filter(pl.col(length_col) < threshold)
    count = too_short.height

    # 2) Print summary of up to first 10 candidate IDs
    print(f"Found {count} CVs with {text_col} < {threshold} chars. Showing up to 10:")
    rows = too_short.select([id_col, length_col]).iter_rows()
    for i, (cid, length) in enumerate(rows):
        if i == 10:
            break
        print(f" {i + 1}. ID {cid}: {length} chars")

    # 3) Sample one CV at random and print full text
    if count > 0:
        # collect candidate IDs and choose one
        ids = too_short.select(id_col).to_series().to_list()
        sample_id = random.choice(ids)
        sample_text = too_short.filter(pl.col(id_col) == sample_id)[text_col][0]
        print(f"\nExample full CV for ID {sample_id}:\n{sample_text!r}")
    else:
        print("\nNo CVs below threshold.")

    return too_short


def plot_length_histogram(
    df: pl.DataFrame,
    text_col: str = "CV_text_anon",
    bin_size: int = 300,
    max_bin: int = 3000,
):
    """ """
    # 1) Compute lengths
    lengths = df.with_columns(
        [
            pl.col(text_col)
            .map_elements(lambda s: len(s.strip()) if s else 0, return_dtype=pl.Int64)
            .alias("length")
        ]
    )

    # 2) Build bin labels
    bin_edges = list(range(0, max_bin + bin_size, bin_size))
    labels = [f"{start}-{start + bin_size}" for start in bin_edges[:-1]]
    labels[-1] = f"{bin_edges[-2]}+"

    def assign_bin(n: int) -> str:
        if n >= max_bin:
            return labels[-1]
        idx = n // bin_size
        return labels[idx]

    binned = lengths.with_columns(
        [
            pl.col("length")
            .map_elements(assign_bin, return_dtype=pl.Utf8)
            .alias("length_bin")
        ]
    )

    bin_counts = (
        binned.group_by("length_bin")
        .agg(pl.count().alias("count"))
        # extract numeric start for sorting
        .with_columns(
            [
                pl.col("length_bin")
                .str.extract(r"^(\d+)", 1)
                .cast(pl.Int64)
                .alias("_start")
            ]
        )
        .sort("_start")
        .drop("_start")
    )

    labels_ordered = bin_counts["length_bin"].to_list()
    counts = bin_counts["count"].to_list()
    plt.figure(figsize=(8, 6))
    plt.bar(labels_ordered, counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("CV Length (characters)")
    plt.ylabel("Number of CVs")
    plt.title(f"Distribution of {text_col} Lengths ({bin_size}-char bins)")
    plt.tight_layout()
    plt.show()


def detect_repetitive_cvs(
    df: pl.DataFrame,
    text_col: str = "CV_text_anon",
    max_repetition: float = 0.5,
) -> pl.DataFrame:
    """
    Identify CVs that—despite having many repeated boilerplate.
    """

    def analyze(text: str) -> Tuple[int, float]:
        if not text:
            return 0, 1.0
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        total = len(lines)
        if total == 0:
            return 0, 1.0
        unique = len(set(lines))
        return total, 1 - unique / total

    annotated = df.with_columns(
        [
            pl.col(text_col)
            .map_elements(lambda s: analyze(s)[0], return_dtype=pl.Int64)
            .alias("n_lines"),
            pl.col(text_col)
            .map_elements(lambda s: analyze(s)[1], return_dtype=pl.Float64)
            .alias("repetition_ratio"),
        ]
    )

    repetitive = annotated.filter(pl.col("repetition_ratio") > max_repetition)
    return repetitive


def detect_vocab_sparsity(
    df: pl.DataFrame,
    text_col: str = "CV_text_anon_clean",
    min_words: int = 20,
    min_ttr: float = 0.1,
) -> pl.DataFrame:
    """
    Identifies CVs that are either too few unique words or too repetitive.
    """

    # 1. Extract raw texts
    texts: List[str] = df.select(text_col).to_series().to_list()

    totals, uniques, ttrs = [], [], []
    for s in texts:
        if not s.strip():
            totals.append(0)
            uniques.append(0)
            ttrs.append(0.0)
            continue
        tokens = s.split()
        total_words = len(tokens)
        unique_words = len({tok.lower() for tok in tokens})
        ttr = unique_words / total_words if total_words else 0.0

        totals.append(total_words)
        uniques.append(unique_words)
        ttrs.append(ttr)

    metrics = pl.DataFrame(
        {"total_words": totals, "unique_words": uniques, "ttr": ttrs}
    )
    annotated = df.with_columns(metrics)

    to_discard = annotated.filter(
        (pl.col("total_words") < min_words)
        | ((pl.col("total_words") >= min_words) & (pl.col("ttr") < min_ttr))
    )
    return to_discard


def filter_placeholder_tails(
    df: pl.DataFrame, text_col: str = "CV_text_anon", char: str = "X", min_run: int = 10
) -> pl.DataFrame:
    """
    Return rows where `text_col` ends with at least `min_run` repetitions of `char`.
    """
    # Compile regex matching the character repeated min_run times at end of string
    pattern: Pattern = re.compile(rf"{re.escape(char)}{{{min_run},}}\s*$")

    return df.filter(
        pl.col(text_col).map_elements(
            lambda s, *_: bool(pattern.search(s)), return_dtype=pl.Boolean
        )
    )


def is_unusual_char(c: str) -> bool:
    ordc = ord(c)
    # allow:
    #  • basic ASCII 0x20–0x7E
    #  • Latin-1 Supplement 0xA0–0xFF (accents)
    #  • General Punctuation 0x2000–0x206F (dashes, quotes…)
    #  • whitespace \n, \r, \t
    return not (
        (0x20 <= ordc <= 0x7E)
        or (0xA0 <= ordc <= 0xFF)
        or (0x2000 <= ordc <= 0x206F)
        or c in "\n\r\t"
    )


def detect_corrupted_cvs(
    df: pl.DataFrame,
    text_col: str = "CV_text_anon_clean",
    max_unusual_frac: float = 0.01,
) -> pl.DataFrame:
    """
    Return rows where the fraction of 'unusual' chars > max_unusual_frac.
    """

    def compute_frac(s: str) -> float:
        if not s:
            return 0.0
        total = len(s)
        bad = sum(1 for c in s if is_unusual_char(c))
        return bad / total

    annotated = df.with_columns(
        [
            pl.col(text_col)
            .map_elements(lambda s, *_: compute_frac(s), return_dtype=pl.Float64)
            .alias("unusual_frac")
        ]
    )

    return annotated.filter(pl.col("unusual_frac") > max_unusual_frac)


def assess_translation_completeness(
    df: pl.DataFrame,
    orig_col: str = "CV_text_anon",
    trans_col: str = "Translated_CV",
) -> pl.DataFrame:
    """ """
    df = df.with_columns(
        [
            pl.col(orig_col)
            .map_elements(lambda s, *_: len(s or ""), return_dtype=pl.Int64)
            .alias("orig_len"),
            pl.col(trans_col)
            .map_elements(lambda s, *_: len(s or ""), return_dtype=pl.Int64)
            .alias("trans_len"),
        ]
    )
    df = df.with_columns(
        # Compute len_ratio and trans_empty
        (pl.col("trans_len") / pl.col("orig_len")).fill_null(0.0).alias("len_ratio")
    )
    return df


# make language detection deterministic
DetectorFactory.seed = 0


def is_this_language(text: str, language: str) -> bool:
    """
    Return True if `text` is detected as the specified `language` code.
    """
    try:
        return detect(text) == language
    except Exception:
        return False
