"""
engine.py  —  Bilingual Hybrid Retrieval & Ranking Engine
==========================================================
Combines three powerful signals to rank schemes for a user:

  1. LEXICAL RELEVANCE (TF-IDF):
     - Dual-index: searches both Marathi & English keyword bags.
     - English intent expansion fallback.
  2. SEMANTIC RELEVANCE (Embeddings):
     - Deep sentence embeddings (SentenceTransformers multilingual model).
     - Matches concepts & intent even with zero shared keywords.
  3. ELIGIBILITY CHECKS (Rules):
     - 9 rule-based eligibility checks (age, gender, state, income,
       occupation, residence, BPL, disability, caste) with central fallback.

Final score is a hybrid blend ensuring top results are both highly on-topic
and strictly qualify for the citizen profile.
"""
import json
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import tokenize, keyword_bag, keyword_bag_en
from semantic import build_or_load_embeddings, semantic_search, get_model

DATA_PATH = "data/schemes_bilingual.json"

# ----------------------------------------------------------------------
# English -> Marathi intent expansion (for Marathi lexical fallback)
# ----------------------------------------------------------------------
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
    """Substitute known English intent phrases for their Marathi equivalents."""
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
    """Load the bilingual scheme records (fallback to normalized if not found)."""
    import os
    if not os.path.exists(path) and os.path.exists("data/schemes_normalized.json"):
        path = "data/schemes_normalized.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Dual TF-IDF + Semantic Index
# ----------------------------------------------------------------------
def build_index(schemes: list):
    """
    Builds:
      1. Marathi TF-IDF index
      2. English TF-IDF index
      3. Semantic dense embeddings (loaded from data/embeddings.npy)
    
    Returns (index_bundle, matrix_mr) for backwards compatibility or full bundle:
    index_bundle = {
        'vec_mr': vec_mr, 'mat_mr': mat_mr,
        'vec_en': vec_en, 'mat_en': mat_en,
        'embeddings': embeddings
    }
    """
    # 1. Marathi TF-IDF index
    bags_mr = [keyword_bag(s) for s in schemes]
    vec_mr = TfidfVectorizer(
        analyzer="word",
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        stop_words=None,
        sublinear_tf=True,
    )
    mat_mr = vec_mr.fit_transform(bags_mr)

    # 2. English TF-IDF index
    bags_en = [keyword_bag_en(s) for s in schemes]
    vec_en = TfidfVectorizer(
        analyzer="word",
        stop_words="english",
        lowercase=True,
        sublinear_tf=True,
    )
    mat_en = vec_en.fit_transform(bags_en)

    # 3. Dense semantic embeddings
    embeddings = build_or_load_embeddings(schemes)

    bundle = {
        "vec_mr": vec_mr,
        "mat_mr": mat_mr,
        "vec_en": vec_en,
        "mat_en": mat_en,
        "embeddings": embeddings,
    }
    return bundle, mat_mr


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
    """Return (eligible, passed_labels, failed_labels, fraction)."""
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
        try:
            amin = float(amin) if amin is not None and str(amin).lower() != "none" else None
        except Exception:
            amin = None
        try:
            amax = float(amax) if amax is not None and str(amax).lower() != "none" else None
        except Exception:
            amax = None

        if amin is not None and age < amin:
            failed.append(f"वय {age} < {int(amin)}")
        elif amax is not None and age > amax:
            failed.append(f"वय {age} > {int(amax)}")
        else:
            passed.append(f"वय {int(amin) if amin else 0}–{int(amax) if amax else '∞'}")

    # 2. Gender
    sg = (scheme.get("eligibility_gender") or "all").lower()
    if gender != "all" and sg not in ("all", gender):
        failed.append(f"लिंग {sg}")
    else:
        passed.append("लिंग ✓")

    # 3. State (a Central scheme: मध्यवर्ती applies to every state)
    ss = scheme.get("state") or "all"
    is_central = "मध्यवर्ती" in ss or "central" in ss.lower()
    if state != "all" and not is_central and ss != state:
        failed.append(f"राज्य {ss}")
    else:
        passed.append("राज्य ✓")

    # 4. Income
    imax = scheme.get("eligibility_income_max")
    income = profile.get("income")
    if imax is not None and str(imax).lower() != "none" and income is not None:
        try:
            imax_val = float(imax)
            if income > imax_val:
                failed.append(f"उत्पन्न {income} > ₹{int(imax_val):,}")
            else:
                passed.append("उत्पन्न ✓")
        except Exception:
            passed.append("उत्पन्न ✓")
    else:
        passed.append("उत्पन्न ✓")

    # 5. Beneficiary type / occupation
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
    bpl_req = scheme.get("eligibility_bpl")
    if isinstance(bpl_req, str):
        bpl_req = bpl_req.lower() == "true"
    if profile.get("bpl") and bpl_req is False:
        failed.append("BPL आवश्यक")
    else:
        passed.append("BPL ✓")

    # 8. Disability
    dis_req = scheme.get("eligibility_disability")
    if isinstance(dis_req, str):
        dis_req = dis_req.lower() == "true"
    if profile.get("disability") and dis_req is False:
        failed.append("दिव्यांग आवश्यक")
    else:
        passed.append("दिव्यांग ✓")

    # 9. Caste
    castes = _norm_castes(scheme)
    if caste != "all" and castes:
        allowed_general = any(str(c).lower() in ("general", "सर्व") for c in castes)
        if not allowed_general and not any(str(c).lower() == caste.lower() for c in castes):
            failed.append(f"जात {', '.join(str(c) for c in castes)}")
        else:
            passed.append("जात ✓")
    else:
        passed.append("जात ✓")

    eligible = len(failed) == 0
    fraction = len(passed) / (len(passed) + len(failed)) if (passed or failed) else 1.0
    return eligible, passed, failed, fraction


# ----------------------------------------------------------------------
# Hybrid Ranking: Lexical (Marathi + English) + Semantic + Eligibility
# ----------------------------------------------------------------------
def rank(schemes, index_bundle, matrix_legacy, query: str, profile: dict, top_k: int = 20):
    """
    Ranks schemes using a tripartite blend:
      - Lexical (TF-IDF): max(Marathi TF-IDF, English TF-IDF)
      - Semantic (Vector): Neural cosine similarity via sentence-transformers
      - Eligibility: Satisfied rules ratio

    Returns list of dicts:
      {scheme, relevance, semantic_score, eligible, fraction, score, reasons}
    """
    raw_query = (query or "").strip()
    n_schemes = len(schemes)

    # Handle both bundle dict and legacy vectorizer argument
    if isinstance(index_bundle, dict):
        vec_mr = index_bundle.get("vec_mr")
        mat_mr = index_bundle.get("mat_mr")
        vec_en = index_bundle.get("vec_en")
        mat_en = index_bundle.get("mat_en")
        embeddings = index_bundle.get("embeddings")
    else:
        vec_mr = index_bundle
        mat_mr = matrix_legacy
        vec_en, mat_en, embeddings = None, None, None

    if raw_query:
        # 1. Marathi lexical relevance (with English->Marathi query expansion)
        query_mr = expand_query(raw_query)
        q_vec_mr = vec_mr.transform([query_mr])
        sims_mr = mat_mr.dot(q_vec_mr.T).toarray().ravel()

        # 2. English lexical relevance
        if vec_en is not None and mat_en is not None:
            q_vec_en = vec_en.transform([raw_query])
            sims_en = mat_en.dot(q_vec_en.T).toarray().ravel()
        else:
            sims_en = np.zeros(n_schemes)

        # Best lexical score between the two languages
        lexical_sims = np.maximum(sims_mr, sims_en)

        # 3. Dense semantic search
        if embeddings is not None:
            sem_sims = semantic_search(raw_query, embeddings)
        else:
            sem_sims = np.zeros(n_schemes)

        # Combined text relevance (40% keyword + 60% semantic understanding)
        text_relevance = 0.40 * lexical_sims + 0.60 * sem_sims
    else:
        lexical_sims = np.zeros(n_schemes)
        sem_sims = np.zeros(n_schemes)
        text_relevance = np.zeros(n_schemes)

    results = []
    for i, s in enumerate(schemes):
        eligible, passed, failed, fraction = check_eligibility(s, profile)
        lex = float(lexical_sims[i])
        sem = float(sem_sims[i])
        rel = float(text_relevance[i])

        if raw_query:
            # Tripartite blend: 30% Lexical + 30% Semantic + 40% Eligibility
            score = 0.30 * lex + 0.30 * sem + 0.40 * fraction
        else:
            score = fraction

        results.append({
            "scheme": s,
            "relevance": round(rel, 4),
            "lexical": round(lex, 4),
            "semantic": round(sem, 4),
            "eligible": bool(eligible),
            "fraction": round(fraction, 2),
            "score": round(score, 4),
            "reasons": failed or passed,
        })

    # Sort eligible schemes first, then by hybrid score
    results.sort(key=lambda r: (r["eligible"], r["score"]), reverse=True)
    return results[:top_k]


# ----------------------------------------------------------------------
# CLI demo: run `python engine.py`
# ----------------------------------------------------------------------
def demo():
    schemes = load_schemes()
    index_bundle, mat_mr = build_index(schemes)
    print("=" * 70)
    print(f"BILINGUAL HYBRID ENGINE READY  |  {len(schemes)} schemes")
    print(f"Marathi Vocab: {index_bundle['mat_mr'].shape[1]} | English Vocab: {index_bundle['mat_en'].shape[1]}")
    print(f"Semantic Embeddings: {index_bundle['embeddings'].shape}")
    print("=" * 70)

    test_cases = [
        ("scholarship for higher education",
         dict(age=20, gender="all", state="महाराष्ट्र", income=150000, occupation="विद्यार्थी")),
        ("महिला व्यवसाय कर्ज",
         dict(age=30, gender="female", state="महाराष्ट्र", income=300000, occupation="उद्योग")),
        ("farmer financial subsidy for agriculture equipment",
         dict(age=45, gender="male", state="all", income=100000, occupation="शेतकरी")),
    ]

    for query, profile in test_cases:
        print("\n" + "#" * 70)
        print(f"QUERY  : {query}")
        print(f"PROFILE: {profile}")
        print("#" * 70)
        for r in rank(schemes, index_bundle, mat_mr, query, profile, top_k=4):
            mark = "✅" if r["eligible"] else "❌"
            s = r["scheme"]
            name_disp = s.get("scheme_name_en") or s.get("scheme_name")
            print(f"  {mark} [Score: {r['score']:.3f} | Lex: {r['lexical']:.3f} | Sem: {r['semantic']:.3f}]")
            print(f"     MR: {s['scheme_name']}")
            print(f"     EN: {s.get('scheme_name_en')}")
            print(f"     State: {s.get('state')} | Reasons: {', '.join(r['reasons'][:2])}")
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo()