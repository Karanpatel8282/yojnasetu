"""
normalize.py  —  Phase 2: Data Normalization
=============================================
Clean the messy `category` and `state` fields in schemes.json so that
every scheme has ONE canonical state and a clean list of canonical
categories. This removes the "silent invisibility" bug where a filter
misses schemes that use a different spelling.

Summary of what we fix:
  * category  : 157 distinct messy strings  ->  ~17 canonical Marathi tags
  * state     : 39 strings, with known pair-variants collapsed to one
"""
import json
import collections
import sys


# ----------------------------------------------------------------------
# 1. Canonical category mapping  (variant -> canonical Marathi tag)
# ----------------------------------------------------------------------
CATEGORY_MAP = {
    # Education (fixes the "शिक्षण आणि शिक्षण" typo)
    "शिक्षण आणि शिकणे": "शिक्षण",
    "शिक्षण आणि शिक्षण": "शिक्षण",
    "शिक्षण": "शिक्षण",
    # Agriculture
    "कृषी, ग्रामीण आणि पर्यावरण": "कृषी",
    "Agriculture, Rural & Environment": "कृषी",
    # Social welfare (samaj/samajik variants)
    "सामाजिक कल्याण आणि सक्षमीकरण": "सामाजिक कल्याण",
    "सामाजिक कल्याण": "सामाजिक कल्याण",
    "समाज कल्याण आणि सक्षमीकरण": "सामाजिक कल्याण",
    "समाज कल्याण": "सामाजिक कल्याण",
    # Health
    "आरोग्य आणि निरोगीपणा": "आरोग्य",
    "आरोग्य आणि कल्याण": "आरोग्य",
    # Women & Child (mul / balak)
    "महिला आणि बालक": "महिला आणि बालक",
    "महिला आणि मूल": "महिला आणि बालक",
    # Skills & employment (kaushalye / kaushal)
    "कौशल्य आणि रोजगार": "कौशल्य व रोजगार",
    "कौशल्ये आणि रोजगार": "कौशल्य व रोजगार",
    # Business
    "व्यवसाय आणि उद्योजकता": "व्यवसाय",
    # Housing
    "गृहनिर्माण आणि निवारा": "गृहनिर्माण",
    # Sports & culture (krida / khel)
    "खेळ आणि संस्कृती": "खेळ आणि संस्कृती",
    "क्रीडा आणि संस्कृती": "खेळ आणि संस्कृती",
    # Tourism
    "प्रवास आणि पर्यटन": "प्रवास आणि पर्यटन",
    # Transport
    "वाहतूक आणि पायाभूत सुविधा": "वाहतूक व पायाभूत सुविधा",
    # Science / IT / Communications (Marathi + English)
    "विज्ञान, आयटी आणि कम्युनिकेशन्स": "विज्ञान, आयटी व कम्युनिकेशन्स",
    "Science, IT & Communications": "विज्ञान, आयटी व कम्युनिकेशन्स",
    # Banking / FSI (Marathi + English)
    "बँकिंग, वित्तीय सेवा आणि विमा": "बँकिंग व विमा",
    "Banking, Financial Services and Insurance": "बँकिंग व विमा",
    "Banking, Financial Services": "बँकिंग व विमा",
    # Utilities / sanitation
    "उपयुक्तता आणि स्वच्छता": "उपयुक्तता व स्वच्छता",
    # Public safety / law (Marathi + English)
    "सार्वजनिक सुरक्षा, कायदा आणि न्याय": "सार्वजनिक सुरक्षा व कायदा",
    "Public Safety, Law & Justice": "सार्वजनिक सुरक्षा व कायदा",
}


# Pass-2: English fragments that appear AFTER the top-level split on ', '.
# The whole-string map above catches combined strings, but the same English
# words also occur as standalone fragments (e.g. "Agriculture") — map those
# onto the same Marathi canonical so English and Marathi unify.
FRAGMENT_MAP = {
    "Agriculture": "कृषी",
    "Rural & Environment": "ग्रामीण व पर्यावरण",
    "Banking": "बँकिंग",
    "Financial Services and Insurance": "वित्तीय सेवा व विमा",
    "Financial Services": "वित्तीय सेवा",
    "Science": "विज्ञान",
    "IT & Communications": "आयटी व कम्युनिकेशन्स",
    "Public Safety": "सार्वजनिक सुरक्षा",
    "Law & Justice": "कायदा व न्याय",
}


def _canon(part: str) -> str:
    """Map one category fragment to its canonical tag (CATEGORY_MAP then
    FRAGMENT_MAP), then normalise the connective 'and':
    'आणि'/'&' -> 'व' so 'ग्रामीण आणि पर्यावरण' and 'ग्रामीण व पर्यावरण' unify."""
    resolved = CATEGORY_MAP.get(part, FRAGMENT_MAP.get(part, part))
    resolved = resolved.replace("आणि", "व").replace(" & ", " व ").replace(" &", " व")
    return " ".join(resolved.split())   # collapse stray double spaces


def normalize_category(raw: str) -> list:
    """Always split a category string on ', ', then map each fragment.
    This way both English ("Agriculture") and Marathi ("कृषी") land on the
    same canonical tag, and multi-category schemes keep every category."""
    if not raw:
        return []
    seen, out = set(), []
    for part in raw.split(", "):
        part = part.strip()
        if not part:
            continue
        canon = _canon(part)                 # CATEGORY_MAP then FRAGMENT_MAP
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


# ----------------------------------------------------------------------
# 2. Canonical state aliases  (variant -> canonical state)
# ----------------------------------------------------------------------
STATE_ALIAS = {
    "तामिळनाडू": "तमिळनाडू",                 # same state, 2 spellings
    "अंदमान आणि निकोबार बेटे": "अंदमान आणि निकोबार",  # suffix variant
    # "मध्यवर्ती" means "Central" - keep as a canonical central tag
}


def normalize_state(raw: str) -> str:
    return STATE_ALIAS.get(raw, raw)


# ----------------------------------------------------------------------
# 3. Main pipeline
# ----------------------------------------------------------------------
def run():
    with open("data/schemes.json", "r", encoding="utf-8") as f:
        schemes = json.load(f)

    changed_cat = 0
    changed_state = 0
    for s in schemes:
        old_cat = s["category"]
        prev_cats = s.get("categories")            # present only on re-runs
        s["categories"] = normalize_category(old_cat)   # NEW clean list
        if prev_cats is None or prev_cats != s["categories"]:
            changed_cat += 1
        s["category"] = ", ".join(s["categories"])          # canonical string

        old_state = s["state"]
        s["state"] = normalize_state(old_state)
        if s["state"] != old_state:
            changed_state += 1

    with open("data/schemes_normalized.json", "w", encoding="utf-8") as f:
        json.dump(schemes, f, ensure_ascii=False, indent=2)

    # ---- Report ----
    print("=" * 70)
    print("NORMALIZATION REPORT")
    print("=" * 70)
    print(f"Total schemes                : {len(schemes)}")
    print(f"Schemes whose category changed: {changed_cat}")
    print(f"Schemes whose state changed   : {changed_state}")
    print()

    # duplicate names
    names = collections.Counter(s["scheme_name"] for s in schemes)
    dups = {k: v for k, v in names.items() if v > 1}
    print(f"Duplicate scheme names ({len(dups)}):")
    for k, v in dups.items():
        print(f"   x{v}  {k}")
    print()

    # canonical categories now in use
    cats = collections.Counter(c for s in schemes for c in s["categories"])
    print(f"Canonical categories now used ({len(cats)}):")
    for k, v in cats.most_common():
        print(f"   {v:4d}  {k}")
    print()

    # canonical states now in use
    sts = collections.Counter(s["state"] for s in schemes)
    print(f"Canonical states now used ({len(sts)}):")
    for k, v in sts.most_common():
        print(f"   {v:4d}  {k}")
    print()
    print("Output written to data/schemes_normalized.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()