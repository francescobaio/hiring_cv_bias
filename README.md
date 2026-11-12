<div align="center">
<h1>Hiring CV bias</h1>

[Francesco Baiocchi](https://github.com/francescobaio), [Leonardo Petrilli](https://github.com/leonardopetrilli)
</div>

## Overview

Recruiting agencies use software to read CVs and to rank candidates. A *parser* extracts a structured profile from
each CV (driving licenses, languages, job titles, skills), and a *matcher* scores the profile against the job offers.
An error or a bias in these steps can exclude qualified candidates.

This repository studies the pipeline of Adecco on anonymized data, in two parts:

1. **Bias of the CV parser** (project of *AI in Industry*). We compare what a proprietary parser extracts with what is
   written in the CV text, and we measure whether its errors differ across gender, macro-region and CV length.
2. **Fairness audit of the selection** (project of *Ethics in Artificial Intelligence*, Module 2). We follow the
   parser's errors down to the selection: do they change who is selected for a job, and do they penalize some groups
   more than others?

<div>
  <img src="https://github.com/user-attachments/assets/fb1caf28-1cf9-4079-bb58-715a57f6f9b7"
       alt="Hiring CV bias app"
       style="display: block;
              width: 100%;
              margin: 50px auto 30px;" />
</div>

> [!NOTE]
> The CV data are anonymized but proprietary, so they are not in the repository and the experiments on them cannot be
> reproduced without them. The unit tests and the toy mitigation example need no data.

The website presents both parts: <https://francescobaio.github.io/hiring_cv_bias/>

## Main results

**Bias of the CV parser.** The parser misses much of what is in the CVs: its recall against our extractors is 0.41 for
driving licenses, 0.66 for languages and 0.55 for job titles. The miss rates are similar across gender and
macro-region.

**Fairness audit.**

* Adecco's match score is a weighted sum of seven components. Our reconstruction reproduces 99.9% of the scores
  exactly.
* The parser's errors reach the selection: correcting them changes the outcome of 8.3% of the (candidate, job) pairs.
* Every group is selected at the same rate (independence holds), and there is no gender gap.
* The decisions on candidates of the South are wrong more often, in both directions (separation and sufficiency
  fail). The cause is mostly in the job lists, not in the parser: southern jobs more often require no skill that
  separates the candidates, so the scores are close to the cut and any noise flips more decisions.
* The usual mitigation techniques have little to act on in the real system, so they are shown on a toy case.

## Repository structure

```text
hiring_cv_bias/
├── bias_detection/              # AI in Industry: errors of the parser
│   ├── fuzzy/                   # exact and semantic matching of job titles (ESCO)
│   └── rule_based/              # regex extractors for licenses and languages,
│                                #   evaluation per group, Streamlit app
├── cleaning/                    # cleaning of the raw CVs
├── exploration/                 # distributions and disparity metrics of the parsed skills
├── hard_soft_skills_labelling/  # labeling of the skills as hard or soft
├── translation/                 # translation of the CVs into English
├── reverse_matching/            # AI Ethics, phase 1: from the raw files to the pairs dataset
│   ├── loader.py                # parser output, false negatives, ESCO labels
│   ├── profiles.py              # three versions of a profile: real (P), parser (O), corrected (C)
│   ├── components.py            # the rules of the score components
│   ├── scoring.py               # outliers, weights, the formula and its exact check
│   ├── experience.py            # estimate of the Experience component with a sentence encoder
│   ├── build.py                 # builds data/ethics/pairs.parquet
│   └── audit.py                 # tables of the data audit
├── fairness/                    # AI Ethics, phases 2 to 4
│   ├── protocol.py              # what counts as a selection: ranking, ties, outcomes
│   ├── metrics.py               # independence, separation, sufficiency, bootstrap
│   ├── robustness.py            # margins to the cut, classes of lists, noise
│   ├── mitigation.py            # toy case: reweighing, Prejudice Remover, equalized odds
│   └── plots.py                 # figures
└── config.py                    # paths of the data

notebooks/
├── ai_in_industry/              # data_cleaning, data_exploration, distributions_analysis, bias_detection
└── ethics/                      # data_audit_and_protocol, propagation, mitigation

tests/                           # unit tests of reverse_matching and fairness
pages/, index.qmd, _quarto.yml   # the website (Quarto)
```

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later. The dependencies of the two parts are
separate extras:

```bash
uv sync --extra ethics        # notebooks/ethics
uv sync --extra ai-industry   # notebooks/ai_in_industry
```

Run the tests, which need no data:

```bash
uv run pytest
```

With the Adecco data in `data/`, the notebooks run in this order:

1. `notebooks/ai_in_industry/`: `data_cleaning`, `data_exploration`, `distributions_analysis`, `bias_detection`. They
   write the cleaned candidates and the parser's false negatives to `data/Adecco_Dataset_cleaned/`.
2. `notebooks/ethics/`: `data_audit_and_protocol`, `propagation`, `mitigation`. The first one builds
   `data/ethics/pairs.parquet`; the same step can be run alone with

   ```bash
   uv run python -m hiring_cv_bias.reverse_matching.build
   ```

   The `mitigation` notebook uses only generated data and runs without the Adecco files.

The website is built with `quarto render` from the outputs stored in the notebooks.

## Development

* `uv run pre-commit install` enables the hooks: ruff (lint and format), mypy and nbstripout.
* GitLab CI and GitHub Actions run `pytest` and `mypy` on every merge request and on `main`. GitHub Actions also
  publishes the website.
