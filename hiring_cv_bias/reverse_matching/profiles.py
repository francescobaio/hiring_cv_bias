"""Candidate profiles as list columns (licenses, languages, regulated, skills, experience)
in three versions: P from the ``Candidates`` sheet, O from the parser output only, C = O plus
the false negatives recovered from the CV text. Items are lower-cased labels in the vocabulary
of the ``Jobs`` sheet (Italian), so the same matching rules apply to every version."""

from collections.abc import Iterable

import polars as pl
from langcodes import Language

PROFILE_FIELDS = ["licenses", "languages", "regulated", "skills", "experience"]
# the fields the parser produces; ``regulated`` and ``skills`` only exist in P
PARSER_FIELDS = ["licenses", "languages", "experience"]

# Candidates sheet column -> profile field
_P_COLUMNS = {
    "drivinglicenses": "licenses",
    "languages": "languages",
    "regulatedtrainings": "regulated",
    "matterknowledges": "skills",
    "experience": "experience",
}

# English ESCO job titles found by the rule-based extractor -> Italian title
# used in the ``experience`` field. Only titles judged unambiguous are kept;
# ``medium``, ``illustrator``, ``extra``, ``host``, ``companion``, ``dresser``,
# ``producer`` and ``animator`` are deliberately excluded as noisy matches.
FN_JOB_TITLE_MAP: dict[str, str] = {
    "cashier": "cassiere",
    "secretary": "segretario",
    "warehouse worker": "magazziniere",
    "waiter": "cameriere",
    "waitress": "cameriere",
    "sales assistant": "commesso",
    "shop assistant": "commesso",
    "bartender": "barman",
    "barista": "barista",
    "accountant": "contabile",
    "promoter": "promoter",
    "chef": "cuoco",
    "cook": "cuoco",
    "babysitter": "baby sitter",
    "tutor": "tutor",
    "receptionist": "receptionist",
    "hostess": "hostess",
    "steward": "steward",
    "electrician": "elettricista",
    "forklift operator": "carrellista",
    "welder": "saldatore",
    "graphic designer": "grafico",
    "bricklayer": "muratore",
    "hairdresser": "parrucchiere",
    "photographer": "fotografo",
    "plumber": "idraulico",
    "carpenter": "falegname",
    "lawyer": "avvocato",
    "security guard": "guardia giurata",
    "office manager": "office manager",
    "tourist guide": "guida turistica",
    "pastry chef": "pasticcere",
    "doorman": "portiere",
    "office clerk": "impiegato",
    "baker": "panettiere",
    "translator": "traduttore",
    "project manager": "project manager",
    "department manager": "responsabile di reparto",
    "notary": "notaio",
    "postman": "postino",
    "personal trainer": "personal trainer",
    "sommelier": "sommelier",
    "management assistant": "assistente di direzione",
    "kitchen assistant": "aiuto cuoco",
    "warehouse manager": "responsabile di magazzino",
    "accounting analyst": "analista contabile",
    "interpreter": "interprete",
    "butcher": "macellaio",
    "fitness instructor": "istruttore di fitness",
    "electrical engineer": "ingegnere elettrico",
    "goldsmith": "orafo",
    "architect": "architetto",
    "room attendant": "cameriere ai piani",
    "journalist": "giornalista",
    "archivist": "archivista",
    "social worker": "assistente sociale",
    "sales manager": "responsabile vendite",
    "visual merchandiser": "visual merchandiser",
    "human resources manager": "responsabile risorse umane",
}

_GENDER_SUFFIX = r"(?i)\s*\((m/f|f/m|other|altro|m/w)\)\s*"


def normalize_item(item: pl.Expr) -> pl.Expr:
    """Lower-case, strip and drop gender suffixes such as ``(m/f)``; null becomes ``""``."""
    return (
        item.fill_null("")
        .str.replace_all(_GENDER_SUFFIX, " ")
        .str.strip_chars()
        .str.to_lowercase()
    )


def split_items(text: pl.Expr, sep: str = ";") -> pl.Expr:
    """Split a ``;``-separated field into normalized, unique, non-empty items."""
    item = normalize_item(pl.element())
    return (
        text.fill_null("")
        .str.split(sep)
        .list.eval(item.filter(item != ""))
        .list.unique(maintain_order=True)
    )


def _italian_language_names(values: Iterable[str], by_code: bool) -> dict[str, str]:
    """Italian name (the ``Jobs`` vocabulary) of each language code or English name;
    values that langcodes does not know are left out."""
    names = {}
    for v in set(values):
        try:
            lang = Language.get(v) if by_code else Language.find(v)
        except (LookupError, ValueError):
            continue
        if lang.is_valid():
            names[v] = lang.display_name("it").lower()
    return names


def build_profile_real(candidates: pl.DataFrame) -> pl.DataFrame:
    """P: the ``Candidates`` sheet as list columns."""
    return candidates.select(
        pl.col("CANDIDATE_ID"),
        *[split_items(pl.col(col)).alias(field) for col, field in _P_COLUMNS.items()],
    )


def _parser_lists(parser: pl.DataFrame) -> pl.DataFrame:
    licenses = (
        parser.filter(pl.col("Skill_Type") == "DRIVERSLIC")
        .with_columns(
            normalize_item(
                pl.col("Skill").str.replace(r"(?i)^driver'?s? licen[cs]e\s*", "")
            ).alias("item")
        )
        .filter(pl.col("item") != "")
        .group_by("CANDIDATE_ID")
        .agg(pl.col("item").unique(maintain_order=True).alias("licenses"))
    )
    # parser names are mostly English; the ones langcodes does not find stay as they are
    lang_rows = parser.filter(pl.col("Skill_Type") == "Language_Skill").with_columns(
        pl.col("Skill").str.to_lowercase().str.strip_chars().alias("item")
    )
    languages = (
        lang_rows.with_columns(
            pl.col("item").replace(
                _italian_language_names(lang_rows["item"], by_code=False)
            )
        )
        .group_by("CANDIDATE_ID")
        .agg(pl.col("item").unique(maintain_order=True).alias("languages"))
    )
    experience = (
        parser.filter(pl.col("Skill_Type") == "Job_title")
        .with_columns(
            normalize_item(pl.coalesce(pl.col("Skill_it"), pl.col("Skill"))).alias(
                "item"
            )
        )
        .filter(pl.col("item") != "")
        .group_by("CANDIDATE_ID")
        .agg(pl.col("item").unique(maintain_order=True).alias("experience"))
    )
    return licenses.join(languages, on="CANDIDATE_ID", how="full", coalesce=True).join(
        experience, on="CANDIDATE_ID", how="full", coalesce=True
    )


def _fill_empty_lists(df: pl.DataFrame, fields: Iterable[str]) -> pl.DataFrame:
    return df.with_columns(
        [pl.col(f).fill_null(pl.lit([], dtype=pl.List(pl.String))) for f in fields]
    )


def build_profile_original(real: pl.DataFrame, parser: pl.DataFrame) -> pl.DataFrame:
    """O: licenses, languages and experience from the parser only; ``regulated`` and
    ``skills`` are copied from P because the parser does not produce them."""
    out = real.select("CANDIDATE_ID", "regulated", "skills").join(
        _parser_lists(parser), on="CANDIDATE_ID", how="left"
    )
    return _fill_empty_lists(out, PARSER_FIELDS).select("CANDIDATE_ID", *PROFILE_FIELDS)


def _fn_lists(
    fn_license: pl.DataFrame, fn_language: pl.DataFrame, fn_job_title: pl.DataFrame
) -> pl.DataFrame:
    lic = (
        fn_license.select("CANDIDATE_ID")
        .unique()
        .with_columns(pl.lit(["b"]).alias("fn_licenses"))
    )
    codes = fn_language.select(pl.col("skill").str.to_lowercase())["skill"]
    lang = (
        fn_language.with_columns(
            pl.col("skill")
            .str.to_lowercase()
            .replace_strict(_italian_language_names(codes, by_code=True), default=None)
            .alias("item")
        )
        .drop_nulls("item")
        .group_by("CANDIDATE_ID")
        .agg(pl.col("item").unique(maintain_order=True).alias("fn_languages"))
    )
    job = (
        fn_job_title.with_columns(
            pl.col("skill")
            .str.to_lowercase()
            .str.strip_chars()
            .replace_strict(FN_JOB_TITLE_MAP, default=None)
            .alias("item")
        )
        .drop_nulls("item")
        .group_by("CANDIDATE_ID")
        .agg(pl.col("item").unique(maintain_order=True).alias("fn_experience"))
    )
    return lic.join(lang, on="CANDIDATE_ID", how="full", coalesce=True).join(
        job, on="CANDIDATE_ID", how="full", coalesce=True
    )


def build_profile_corrected(
    original: pl.DataFrame,
    fn_license: pl.DataFrame,
    fn_language: pl.DataFrame,
    fn_job_title: pl.DataFrame,
) -> pl.DataFrame:
    """C: O plus the recovered false negatives. A recovered generic license is recorded as
    class B; language codes are named with langcodes, titles with ``FN_JOB_TITLE_MAP``
    (unmapped titles are ignored)."""
    merged = _fill_empty_lists(
        original.join(
            _fn_lists(fn_license, fn_language, fn_job_title),
            on="CANDIDATE_ID",
            how="left",
        ),
        ["fn_licenses", "fn_languages", "fn_experience"],
    )
    return merged.select(
        "CANDIDATE_ID",
        pl.col("licenses")
        .list.concat(pl.col("fn_licenses"))
        .list.unique(maintain_order=True),
        pl.col("languages")
        .list.concat(pl.col("fn_languages"))
        .list.unique(maintain_order=True),
        "regulated",
        "skills",
        pl.col("experience")
        .list.concat(pl.col("fn_experience"))
        .list.unique(maintain_order=True),
    )
