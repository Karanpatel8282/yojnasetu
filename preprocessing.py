"""
preprocessing.py  —  Phase 3: Marathi Text Preprocessing
==========================================================
Pure-NLP layer. Turns raw scheme text into clean, indexable tokens so that
the Phase 4 retrieval engine can match a user query against a scheme.

Pipeline for each field:
    raw text -> normalize -> tokenize -> lowercase/clean -> remove stopwords
"""
import re
import json
import sys

# ----------------------------------------------------------------------
# Marathi stopwords (function words carry no meaning for retrieval).
# Tailored set of common Marathi connectors / pronouns / auxiliaries.
# ----------------------------------------------------------------------
MARATHI_STOPWORDS = set("""
मी तू तो ती ते आम्ही तुम्ही आपण हा ही हे या त्या तो ती
आहे आहेत आहोत होते होतं झाले झालं असतो असते होईल
आणि व नव्हे नाही नका नको
पण तर म्हणून किंवा तसेच सुद्धा बरोबर पासून साठी करीता की
असा अशी असे अशा त्याचा त्यांचा त्याच्या तिचा तिच्या ज्याचा ज्यांचा
याचा यांचा कोणी सर्व प्रत्येक काही काहीही अनेक कोणती कोणते किती
नंतर आधी अगदी अर्थातच मात्र फक्त केवळ
एक
""".split())

# Tokens or token-letters that can never be useful retrieval keywords
NON_KEYWORD = re.compile(r"^[\W_]+$")          # pure punctuation
SHORT_TOKEN = 2                                # keep tokens of length >= 2

SAFE_PATTERN = re.compile(r"[^\wऀ-ॿ]", re.UNICODE)


# ----------------------------------------------------------------------
# Light rule-based Marathi stemmer (suffix stripping).
# Marathi is morphologically rich: शिष्यवृत्तीच्या / शिष्यवृत्तीला reduce to
# the base शिष्यवृत्ती. We strip the most common postpositions and oblique
# case markers so query and document tokens unify (recall gain).
#
# CAUTION: only clearly-marker suffixes are stripped. Bare single/double
# letters like 'ला', 'ने', 'त' are NOT stripped, because they collide with
# real word roots (महिला must not become महि). That keeps stemming safe.
# ----------------------------------------------------------------------
STEM_SUFFIXES = sorted({
    "ण्यासाठी", "ण्याकरता", "ण्याने", "ण्यात", "ण्याच्या", "ण्याचे",
    "ण्याची", "ण्याचा",
    "ासाठी", "ांसाठी", "साठी",
    "च्याकडून", "कडून", "द्वारे", "प्रमाणे", "मुळे", "बद्दल", "पर्यंत",
    "ांप्रमाणे",
    "ांनी", "ांच्या", "ांची", "ांचे", "ांचा",
    "ाने", "च्या", "ची", "चे", "चा", "ांना",
}, key=len, reverse=True)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def stem(token: str) -> str:
    """Strip the first matching Marathi suffix (longest-first), if any."""
    if len(token) <= 4 or not DEVANAGARI.search(token):
        return token
    for sfx in STEM_SUFFIXES:
        if token.endswith(sfx) and len(token) - len(sfx) >= 3:
            return token[: -len(sfx)]
    return token


def _normalize_casing(tok: str) -> str:
    """Devanagari has no case; lowercase English loan tokens for consistency."""
    return tok.lower()


def tokenize(text: str) -> list:
    """Full preprocessing: normalize -> split -> clean -> drop stopwords.

    Returns a list of keyword tokens ready for indexing / matching.
    """
    if not text:
        return []

    # 1. Normalize connective variants & whitespace
    text = text.replace("  ", " ")

    words = re.split(r"\s+", text.strip())
    out = []
    for w in words:
        # keep only word characters + Devanagari range; strip punctuation
        cleaned = SAFE_PATTERN.sub("", w)
        if not cleaned:
            continue
        cleaned = _normalize_casing(cleaned)
        if NON_KEYWORD.match(cleaned):
            continue
        if len(cleaned) < SHORT_TOKEN:
            continue
        if cleaned in MARATHI_STOPWORDS:
            continue
        out.append(stem(cleaned))           # reduce to base form
    return out


def scheme_tokens(scheme: dict) -> list:
    """Concatenate a scheme's important fields into one token bag."""
    fields = [
        scheme.get("scheme_name", ""),
        scheme.get("category", ""),
        scheme.get("state", ""),
        scheme.get("description", ""),
        scheme.get("benefits", ""),
        scheme.get("beneficiary_type", ""),
    ]
    return tokenize(" ".join(fields))


def keyword_bag(scheme: dict) -> str:
    """A single space-joined string of a scheme's tokens (for TF-IDF)."""
    return " ".join(scheme_tokens(scheme))


# ----------------------------------------------------------------------
# Demo / self-test: run `python preprocessing.py` to see it work
# ----------------------------------------------------------------------
def demo():
    with open("data/schemes_normalized.json", "r", encoding="utf-8") as f:
        schemes = json.load(f)

    print("=" * 70)
    print("PREPROCESSING DEMO  (first 5 schemes)")
    print("=" * 70)

    for s in schemes[:5]:
        print(f"\n[{s['scheme_id']}] {s['scheme_name']}")
        raw = (s.get("description") or "")[:160].replace("\n", " ")
        print(f"  RAW : {raw}")
        toks = scheme_tokens(s)
        print(f"  TOKS({len(toks)}): {' '.join(toks[:18])}{' …' if len(toks) > 18 else ''}")

    # Stopword demonstration
    print("\n" + "=" * 70)
    print("STOPWORD DEMO")
    print("=" * 70)
    probe = "मी विद्यार्थी आहे आणि मला शिष्यवृत्ती मिळेल का आणि ती देखील"
    print(" QUERY:", probe)
    print(" TOKENS:", tokenize(probe))
    print("  -> 'मी','आहे','आणि','ती' dropped as stopwords; content words विद्यार्थी शिष्यवृत्ती मिळेल kept")

    # Coverage
    all_toks = set()
    for s in schemes:
        all_toks.update(scheme_tokens(s))
    print(f"\nCorpus: {len(schemes)} schemes, vocabulary of {len(all_toks)} unique tokens")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo()