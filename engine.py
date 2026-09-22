"""
engine.py  —  Phase 4: Retrieval & Ranking Engine
==================================================
Combines two signals to rank schemes for a user:

  1. RELEVANCE : TF-IDF + cosine similarity between the user's free-text
                 query and each scheme's keyword bag (name/desc/benefits/...).
  2. ELIGIBILITY : rule-based check of the user profile against each
                 scheme's structured eligibility fields.

Final score = blend of the two, so the top result is BOTH on-topic AND
something the user actually qualifies for. Every result carries a readable
"why eligible" explanation.
"""
import json
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import tokenize, keyword_bag

DATA_PATH = "data/schemes_normalized.json"

# ----------------------------------------------------------------------
# English -> Marathi intent expansion
# ----------------------------------------------------------------------
# The corpus is Marathi-only, so an English query would miss everything.
# We map common English intent words to their Marathi equivalents and
# substitute them into the query BEFORE vectorising, so "farmer loan"
# becomes "शेतकरी कर्ज" and matches the index. Phrases are tried
# longest-first so "senior citizen" beats "senior". Cheap (<= ~40 terms).
EN_MR = {
    # scholarships / education
    "scholarship": "शिष्यवृत्ती", "student": "विद्यार्थी", "education": "शिक्षण",
    "school": "शाळा", "college": "महाविद्यालय", "hostel": "वसतिगृह",
    "laptop": "लॅपटॉप", "computer": "संगणक", "internship": "इंटर्नशिप",
    "training": "प्रशिक्षण", "skill": "कौशल्य", "skills": "कौशल्य",
    # agriculture / rural
    "farmer": "शेतकरी", "agriculture": "कृषी", "crop": "पीक", "seed": "बियाणे",
    "irrigation": "सिंचन", "solar": "सौर", "farm": "शेत", "rural": "ग्रामीण",
    # finance / loans
    "loan": "कर्ज", "loan waiver": "कर्जमाफी", "subsidy": "अनुदान",
    "grant": "अनुदान", "pension": "पेन्शन", "insurance": "विमा",
    "mudra": "मुद्रा", "startup": "स्टार्टअप", "business": "उद्योग",
    "enterprise": "उद्योजकता", "self employment": "स्वरोजगार",
    # demographics / welfare
    "woman": "महिला", "women": "महिला", "female": "महिला", "girl": "मुलगी",
    "widow": "विधवा", "elderly": "ज्येष्ठ नागरिक", "senior": "ज्येष्ठ",
    "senior citizen": "ज्येष्ठ", "old age": "ज्येष्ठ", "worker": "कामगार",
    "labour": "कामगार", "minority": "अल्पसंख्यांक", "tribal": "आदिवासी",
    "disability": "दिव्यांग", "disabled": "अपंग", "unemployed": "बेरोजगार",
    "unemployment": "बेरोजगार", "housing": "गृहनिर्माण", "house": "घर",
    "marriage": "लग्न", "health": "आरोग्य", "medicines": "औषधे",
}


def expand_query(query: str) -> str:
    """Substitute known English intent phrases for their Marathi equivalents.

    Unrecognised tokens (and Marathi input) pass through unchanged, so this
    is safe to call on every query.
    """
    import re
    q = query.strip()
    if not q:
        return q
    for en, mr in sorted(EN_MR.items(), key=lambda kv: -len(kv[0])):
        q = re.sub(rf"\b{re.escape(en)}\b", mr, q, flags=re.IGNORECASE)
    return q


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_schemes(path: str = DATA_PATH) -> list:
    """Load the normalized scheme records."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# TF-IDF index (fit once, reuse for every query)
# ----------------------------------------------------------------------
def build_index(schemes: list):
    """Build a TF-IDF vectorizer + document-term matrix over scheme keyword bags.

    Returns (vectorizer, matrix) where matrix[i] is scheme i's tf-idf vector.
    The vectorizer is reused later to transform a user query into the same
    feature space.
    """
    bags = [keyword_bag(s) for s in schemes]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=tokenize,        # our Marathi preprocessing pipeline
        preprocessor=None,
        lowercase=False,           # we handle casing ourselves
        token_pattern=None,        # custom tokenizer, so disable built-in regex
        stop_words=None,           # we removed stopwords in tokenize()
        sublinear_tf=True,         # log(1+tf): dampen very frequent terms
    )
    matrix = vectorizer.fit_transform(bags)          # shape (n_schemes, vocab)
    return vectorizer, matrix


# ----------------------------------------------------------------------
# Eligibility rules
# ----------------------------------------------------------------------
def _norm_castes(scheme):
    """Parse the eligibility_caste JSON string into a list."""
    raw = scheme.get("eligibility_caste") or ""
    if isinstance(raw, list):
        return raw
    try:
        import json as _j
        val = _j.loads(raw)
        return val if isinstance(val, list) else [val]
    except Exception:
        return [raw] if raw else []


def check_eligibility(scheme: dict, profile: dict):
    """Return (eligible, passed_labels, failed_labels).

    `eligible` is True when the profile passes every rule the scheme imposes.
    Each label is a short Marathi string, e.g. "वय 18–45 ✓".
    """
    passed, failed = [], []
    gender = profile.get("gender", "all")
    state = profile.get("state", "all")
    caste = profile.get("caste", "all")
    residence = profile.get("residence", "both")
    occupation = profile.get("occupation", "all")

    # 1. Age
    amin, amax = scheme.get("eligibility_age_min"), scheme.get("eligibility_age_max")
    age = profile.get("age")
    if age is not None:
        if amin is not None and age < amin:
            failed.append(f"वय {age} < {amin}")
        elif amax is not None and age > amax:
            failed.append(f"वय {age} > {amax}")
        else:
            passed.append(f"वय {amin or 0}–{amax or '∞'}")

    # 2. Gender
    sg = (scheme.get("eligibility_gender") or "all").lower()
    if gender != "all" and sg not in ("all", gender):
        failed.append(f"लिंग {sg}")
    else:
        passed.append("लिंग ✓")

    # 3. State (a Central scheme: मध्यवर्ती applies to every state)
    ss = scheme.get("state") or "all"
    is_central = "मध्यवर्ती" in ss
    if state != "all" and not is_central and ss != state:
        failed.append(f"राज्य {ss}")
    else:
        passed.append("राज्य ✓")

    # 4. Income
    imax = scheme.get("eligibility_income_max")
    income = profile.get("income")
    if imax is not None and income is not None and income > imax:
        failed.append(f"उत्पन्न {income} > ₹{imax:,}")
    else:
        passed.append("उत्पन्न ✓")

    # 5. Beneficiary type / occupation
    #   'सर्व', 'वैयक्तिक' (Individual), 'all', 'general' are GENERIC types
    #   that restrict no one - only a specific type (विद्यार्थी, शेतकरी, ...)
    #   is a real constraint.
    bt = (scheme.get("beneficiary_type") or "").lower()
    generic = {"", "सर्व", "all", "वैयक्तिक", "individual", "general", "कोणीही"}
    is_generic = bt in generic
    if occupation != "all" and not is_generic and occupation.lower() not in bt:
        failed.append(f"व्यवसाय {bt}")
    else:
        passed.append("व्यवसाय ✓")

    # 6. Residence
    sr = (scheme.get("eligibility_residence") or "both").lower()
    if residence != "both" and sr != "both" and sr != residence:
        failed.append(f"निवास {sr}")
    else:
        passed.append("निवास ✓")

    # 7. BPL
    if profile.get("bpl") and scheme.get("eligibility_bpl") is False:
        failed.append("BPL आवश्यक")
    else:
        passed.append("BPL ✓")

    # 8. Disability
    if profile.get("disability") and scheme.get("eligibility_disability") is False:
        failed.append("दिव्यांग आवश्यक")
    else:
        passed.append("दिव्यांग ✓")

    # 9. Caste (scheme requires a specific caste list)
    castes = _norm_castes(scheme)
    if caste != "all" and castes:
        allowed_general = any(c.lower() in ("general", "सर्व") for c in castes)
        if not allowed_general and not any(c.lower() == caste.lower() for c in castes):
            failed.append(f"जात {', '.join(castes)}")
        else:
            passed.append("जात ✓")
    else:
        passed.append("जात ✓")

    eligible = len(failed) == 0
    fraction = len(passed) / (len(passed) + len(failed)) if (passed or failed) else 1.0
    return eligible, passed, failed, fraction


# ----------------------------------------------------------------------
# Combined ranking
# ----------------------------------------------------------------------
def rank(schemes, vectorizer, matrix, query: str, profile: dict, top_k: int = 20):
    """Rank schemes by blended relevance + eligibility.

    Returns a list of dicts: {scheme, relevance, eligible, fraction,
    score, reasons}.
    """
    # English intent -> Marathi, so the query aligns with the Marathi index.
    query = expand_query(query)

    # -- relevance via cosine similarity --
    if query and query.strip():
        q_vec = vectorizer.transform([query])
        sims = matrix.dot(q_vec.T).toarray().ravel()     # (n_schemes,)
    else:
        sims = np.zeros(len(schemes))

    results = []
    for i, s in enumerate(schemes):
        eligible, passed, failed, fraction = check_eligibility(s, profile)
        cos = float(sims[i])
        # blend: content relevance + how fully the profile satisfies eligibility
        if query and query.strip():
            score = 0.5 * cos + 0.5 * fraction
        else:
            score = fraction
        results.append({
            "scheme": s,
            "relevance": round(cos, 4),
            "eligible": bool(eligible),
            "fraction": round(fraction, 2),
            "score": round(score, 4),
            "reasons": failed or passed,     # show blockers first, else satisfied
        })

    # eligible & in-topic schemes first, then by score
    results.sort(key=lambda r: (r["eligible"], r["score"]), reverse=True)
    return results[:top_k]


# ----------------------------------------------------------------------
# CLI demo: run `python engine.py` to see ranked results
# ----------------------------------------------------------------------
def demo():
    schemes = load_schemes()
    vectorizer, matrix = build_index(schemes)
    print("=" * 70)
    print(f"ENGINE READY  |  {len(schemes)} schemes  |  "
          f"vocab {matrix.shape[1]} terms")
    print("=" * 70)

    queries = [
        ("शिष्यवृत्ती विद्यार्थी",
         dict(age=20, gender="all", state="महाराष्ट्र",
              income=150000, occupation="विद्यार्थी",
              residence="both", bpl=False, disability=False, caste="all")),
        ("कर्ज महिला उद्योग",
         dict(age=30, gender="female", state="महाराष्ट्र",
              income=300000, occupation="उद्योग",
              residence="both", bpl=False, disability=False, caste="OBC")),
    ]

    for query, profile in queries:
        print("\n" + "#" * 70)
        print(f"QUERY  : {query}")
        print(f"PROFILE: {profile}")
        print("#" * 70)
        for r in rank(schemes, vectorizer, matrix, query, profile, top_k=5):
            mark = "✅" if r["eligible"] else "❌"
            print(f"  {mark} [{r['score']:.3f} | rel {r['relevance']:.3f}] "
                  f"{r['scheme']['scheme_name']}  ({r['scheme']['state']})")
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo()