# 🏛️ YojanaSetu — शासकीय योजना शिफारस व शोध प्रणाली

**A bilingual Marathi NGO scheme-search & recommendation engine that ranks the most relevant Indian government schemes for a user's eligibility profile.**

Built on a real corpus of **1,000 government schemes** spanning all **37 Indian states/UTs plus central**, YojanaSetu fuses free-text relevance (TF-IDF + cosine similarity) with **9 rule-based eligibility checks**, and explains *why* each scheme matches — in both English and मराठी.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results & Insights](#results--insights)
- [Future Scope](#future-scope)
- [Author](#author)

---

## Overview

A scheme register isn't really a list — it's a set of *conditions*. A scheme's usefulness for a given person depends not just on what it's called, but on whether that person's age, income, state, caste, and occupation actually qualify for it. Flipping through 1,000 spreadsheets to find the handful you qualify for is nearly impossible; asking a retrieval engine to check your whole profile is what hybrid search is built for.

**YojanaSetu** analyzes government schemes and user eligibility profiles using a combination of:

- **Marathi NLP preprocessing** — custom Devanagari-aware tokenizer, stopword removal, and light suffix-stripping stemmer
- **Free-text relevance** — TF-IDF (`sublinear_tf`) + cosine similarity over each scheme's text "bag"
- **English→Marathi intent expansion** — English queries are translated into the Marathi index so both languages align
- **Rule-based eligibility** — 9 explainable checks (age, gender, state, income, occupation, residence, BPL, disability, caste) with a central-scheme fallback
- **Interactive visualization** (Streamlit) — a dark, premium bilingual dashboard that surfaces ranked, explained results in seconds

The result is a system that answers a question a plain list never could — *which schemes is this specific person actually eligible for?*

## Problem Statement

A large number of Indian government welfare schemes exist across states and ministries, many of them described in Marathi with messy, inconsistent categories and state tags. Searching them with only traditional keyword matching makes it difficult to see which ones apply to a particular citizen.

This project builds an NLP pipeline that processes the scheme corpus end-to-end and surfaces answers such as:

- Schemes relevant to a free-text Marathi (or English) query
- Schemes a specific eligibility profile actually qualifies for
- Ranked, blended results balancing "on-topic" with "eligible"
- A readable *why-eligible* explanation on every recommendation
- Category and state-level breakdowns of the whole corpus

The project combines classic information retrieval with rule-based eligibility to recommend from both angles.

## Objectives

1. Normalize a messy 1,000-scheme corpus onto a canonical category and state tag set
2. Unify Marathi and English category spellings into a single bilingual vocabulary
3. Build a Devanagari-safe tokenizer for Marathi scheme text
4. Remove Marathi stopwords and apply a light rule-based stemmer
5. Vectorize each scheme as a TF-IDF text bag
6. Expand English queries into Marathi before retrieval
7. Score each query–scheme pair by cosine relevance
8. Define 9 eligibility rules derived from the scheme fields
9. Evaluate each scheme against a user's eligibility profile
10. Blend relevance and eligibility into a single explainable ranking
11. Sort results eligible-first, then by blended score
12. Present all results through a bilingual Streamlit dashboard

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Framework | **Streamlit** | Interactive bilingual web application layer |
| Retrieval | **scikit-learn TF-IDF** | `sublinear_tf` vectorization + cosine similarity |
| Numerical ops | **NumPy** | Sparse-matrix dot products for relevance scoring |
| Data handling | **Pandas** | Corpus loading, normalization, and dashboard datasets |
| Preprocessing | **Custom Python** | Marathi tokenizer, stopwords, light stemmer, intent expansion |
| Storage | **JSON** | Cleaned 1,000-scheme dataset on disk |
| Language | **Python** | End-to-end pipeline implementation |

## Architecture

```text
              schemes.json (1,000 raw schemes)
                           |
                           v
                 [normalize.py]  Phase 2
                   157 to 19 categories · 39 to 37 states
                           |
                           v
               schemes_normalized.json
                           |
                           v
              [preprocessing.py]  Phase 3
              tokenize / stopwords / stem (Marathi)
                           |
                           v
               [engine.py]  Phase 4
                    TF-IDF index
                           |
          +------------------+------------------+
          |                                     |
          v                                     v
     cosine(query, scheme)             9 eligibility rules
      (English→Marathi first)          age·gender·state·income
                                       occupation·residence·BPL
                                       disability·caste
          +------------------+------------------+
                           |
                           v
                 blend: 0.5·relevance + 0.5·eligibility
                           |
                           v
                  ranked, eligible-first results
                           |
                           v
               [app.py]  Streamlit UI  (EN / मराठी)
```

**Pipeline stages, in short:**

1. **Normalize** — messy category/state strings are mapped onto canonical bilingual tags (157→19 categories, 39→37 states)
2. **Preprocess** — Marathi tokenizer strips punctuation, drops stopwords, and light-stems oblique suffix forms
3. **Index** — each scheme becomes a TF-IDF text bag (`sublinear_tf` cosine)
4. **Rank** — English queries expand into Marathi, then cosine relevance is computed against the index
5. **Check** — the user's eligibility profile is run through all 9 rules, each producing a pass/fail explanation
6. **Blend** — `0.5·relevance + 0.5·eligibility` ranks results eligible-first
7. **Visualize** — every result is served through the bilingual Streamlit dashboard

## Dashboard

The dashboard is organized into a single page with two focused tabs, wrapped in a premium dark UI:

| Page | What it shows |
|---|---|
| 🔎 **Recommendations** | Hero search + quick-search pills, 19-category filter, top-k slider, and ranked result cards — each with an eligibility badge (पात्र/अपात्र), match-percentage bar, and *why-eligible* reasons with a detail expander |
| 📊 **Explore Dataset** | Schemes-by-category and top-states bar charts, plus a full browsable dataframe of all 1,000 schemes |

The sidebar hosts a bilingual English/मराठी toggle and the full eligibility profile — age, gender, state, income, caste (SC/ST/OBC/EWS), occupation, residence, BPL, and disability.

Run it with:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
scheme-retrieval-v2/
│
├── app.py              # Streamlit UI — bilingual dashboard (run this)
├── engine.py           # Retrieval + eligibility + ranking engine
├── preprocessing.py    # Marathi tokenizer, stopwords, light stemmer
├── normalize.py        # Data-normalization script (Phase 2)
├── eval.py             # precision@K / recall@K evaluation (Phase 6)
│
├── data/
│   ├── schemes.json              # raw 1,000-scheme corpus (Phase 1 input)
│   └── schemes_normalized.json   # cleaned, canonically-tagged dataset
│
├── .streamlit/
│   └── config.toml     # dark theme — emerald accent, Inter font
│
├── requirements.txt
└── README.md
```

**What lives where:**

- **`app.py`** — the Streamlit app: bilingual UI strings, sidebar eligibility profile, search, result cards, and the Explore tab (reads the engine's cached index)
- **`engine.py`** — the core ranking logic: TF-IDF build, `check_eligibility` (9 rules), and blended `rank()`; also an optional CLI `demo()` for sample queries
- **`preprocessing.py`** — the NLP layer: Devanagari-safe `tokenize`, curated `MARATHI_STOPWORDS`, and light `STEM_SUFFIXES` stemmer that feed the TF-IDF tokenizer
- **`normalize.py`** — Phase 2 cleanup: maps 157 messy categories onto 19 canonical tags and 39 state variants onto 37, appending a `categories` list per record
- **`eval.py`** — Phase 6 evaluation: precision@K / recall@K over a hand-labeled gold set (7 queries × relevant schemes)
- **`data/schemes.json`** — the raw, uncleaned corpus
- **`data/schemes_normalized.json`** — the cleaned dataset the engine and dashboard read from
- **`.streamlit/config.toml`** — the dark-premium theme (emerald `#10b981` accent, deep-navy background, Inter font)

## Getting Started

### Prerequisites

- Python 3.9+ (3.10+ recommended)
- pip / virtualenv
- An internet connection for Streamlit's default browser launch

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/scheme-retrieval-v2.git
cd scheme-retrieval-v2
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. (Optional) Regenerate the cleaned dataset

The normalized dataset is already committed, so you can skip this. To rebuild it from the raw corpus (or re-run the normalization report):

```bash
python normalize.py
```

## Usage

Run the app:

```bash
# 1. Launch the dashboard
python -m streamlit run app.py

# 2. (or) run the other scripts directly
python normalize.py     # regenerate the cleaned dataset + report
python engine.py        # print ranked results for 2 sample queries (CLI)
python eval.py          # precision@K / recall@K on the gold set
python preprocessing.py # demo of tokenizer / stopwords / stemmer
```

The browser opens at `http://localhost:8501`.

> 💡 **Tip:** Use `python -m streamlit` (not bare `streamlit`) if the CLI is not on your PATH. Set the sidebar language to मराठी to switch the whole UI.

## Dataset

This project uses a self-curated corpus of **1,000 real Indian government schemes**, all described in Marathi:

- **`schemes.json`** — the raw corpus: scheme names, descriptions, benefits, eligibility criteria, application process, and documents, with messy category/state strings
- **`schemes_normalized.json`** — the cleaned dataset: 19 canonical categories, 37 states/UTs + central, and a per-record `categories` list

Each record carries 19 fields including the eligibility rule inputs (`eligibility_gender`, `eligibility_caste`, `eligibility_residence`, `eligibility_disability`, `eligibility_bpl`, `eligibility_age_min/max`, `eligibility_income_max`, `beneficiary_type`).

## How the ranking works

Each scheme becomes a text "bag" (name + category + state + description + benefits + beneficiary type), preprocessed with the Marathi pipeline and vectorized with TF-IDF (`sublinear_tf`). For a query and profile:

```text
relevance     = cosine( query_tfidf , scheme_tfidf )   # English query expanded to Marathi first
eligibility   = fraction of the scheme's 9 rules the profile satisfies
   score      = 0.5·relevance + 0.5·eligibility        (when a query is given)
   score      = eligibility                            (browse mode, no query)
```

Schemes are shown **eligible-first**, then by score, with pass/fail reasons printed underneath — reading e.g. `वय 18–45 ✓ · राज्य ✓ · उत्पन्न ₹3,00,000 → ✓`.

### The 9 eligibility rules

age · gender · state (with **मध्यवर्ती central fallback** — central schemes apply to every state) · income · occupation / beneficiary type · residence · BPL · disability · caste.

## Results & Insights

On a gold set of **7 queries × relevant schemes**, the blended engine scores:

| Metric       | Blended engine |
|--------------|----------------|
| Precision@5  | **0.77**       |
| Recall@5     | **0.96**       |
| Precision@10 | 0.39           |
| Recall@10    | **0.96**       |

The blended (relevance + eligibility) ranking closely tracks the content-only (relevance) baseline (R@10: 0.96 vs **1.00**) — i.e. adding the eligibility signal **reorders results without sacrificing recall**, proving the "why eligible" layer is explainable *and* faithful.

## NLP techniques used

- **Data cleaning / normalization** — canonical category & state tag set that unifies Marathi and English spellings (`तामिळनाडू`→`तमिळनाडू`, `Agriculture`→`कृषी`, `शिक्षण आणि शिक्षण`→`शिक्षण`, …)
- **Tokenization** — whitespace splitting with Devanagari-safe cleaning (`[^\wऀ-ॿ]`)
- **Stopword removal** — curated Marathi function-word list (~50–70 tokens)
- **Light stemming** — rule-based suffix stripping for Marathi oblique/postpositional forms (e.g. `शिष्यवृत्तीच्या`→`शिष्यवृत्ती`) with safety guardrails
- **Intent expansion** — English→Marathi substitution (e.g. `farmer loan`→`शेतकरी कर्ज`) so both languages hit the same index
- **TF-IDF + cosine similarity** — `scikit-learn`, `sublinear_tf`
- **Fused relevance + rules** — explainable hybrid scoring

## Future Scope

- Add cross-state scheme matching for schemes currently tagged only to single states
- Expand the corpus with English and Hindi scheme descriptions for multilingual retrieval
- Add voice input for the search bar (Marathi speech-to-text)
- Introduce semantic embeddings (e.g. sentence-transformers / IndicBERT) alongside TF-IDF
- Deploy the dashboard for public access behind a free domain or Streamlit Cloud

## Author

**Karan**
Final-year Computer Engineering student, University of Mumbai
Project area: Natural Language Processing, Information Retrieval, and Machine Learning

---

*Built with Python, scikit-learn, and Streamlit.*