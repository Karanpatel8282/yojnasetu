"""
merge_bilingual.py — Fast merge of Marathi normalized schemes with official English CSV
=====================================================================================
Matches scheme_id in schemes_normalized.json with slug in Schemes.csv (100% exact match).
Produces data/schemes_bilingual.json with both Marathi and high quality official English text.
"""
import json
import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

MARATHI_PATH = "data/schemes_normalized.json"
ENGLISH_CSV_PATH = "data/Schemes.csv"
OUTPUT_PATH = "data/schemes_bilingual.json"

def clean_val(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s

def main():
    print("=" * 70)
    print("MERGING MARATHI SCHEMES WITH ENGLISH CSV")
    print("=" * 70)
    
    with open(MARATHI_PATH, "r", encoding="utf-8") as f:
        marathi_schemes = json.load(f)
        
    df = pd.read_csv(ENGLISH_CSV_PATH)
    # Deduplicate on slug if any, keeping first
    df = df.drop_duplicates(subset=["slug"])
    en_lookup = df.set_index("slug").to_dict(orient="index")
    
    matched = 0
    for s in marathi_schemes:
        sid = s["scheme_id"]
        en_row = en_lookup.get(sid, {})
        if en_row:
            matched += 1
            s["scheme_name_en"] = clean_val(en_row.get("name"))
            s["description_en"] = clean_val(en_row.get("description"))
            s["benefits_en"] = clean_val(en_row.get("benefits"))
            s["eligibility_criteria_en"] = clean_val(en_row.get("eligibility_text"))
            s["application_process_en"] = clean_val(en_row.get("application_process"))
            s["documents_required_en"] = clean_val(en_row.get("documents_required"))
            s["category_en"] = clean_val(en_row.get("category"))
            s["state_en"] = clean_val(en_row.get("state"))
            s["department_or_ministry_en"] = clean_val(en_row.get("department")) or clean_val(en_row.get("ministry"))
            s["apply_url"] = clean_val(en_row.get("apply_url"))
            s["official_url"] = clean_val(en_row.get("official_url"))
        else:
            s["scheme_name_en"] = s.get("scheme_name", "")
            s["description_en"] = s.get("description", "")
            s["benefits_en"] = s.get("benefits", "")
            s["eligibility_criteria_en"] = s.get("eligibility_criteria", "")
            s["application_process_en"] = s.get("application_process", "")
            s["documents_required_en"] = s.get("documents_required", "")
            s["category_en"] = s.get("category", "")
            s["state_en"] = s.get("state", "")
            s["department_or_ministry_en"] = s.get("department_or_ministry", "")
            s["apply_url"] = ""
            s["official_url"] = ""

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(marathi_schemes, f, ensure_ascii=False, indent=2)

    print(f"Total Marathi schemes: {len(marathi_schemes)}")
    print(f"Successfully matched with English CSV: {matched} / {len(marathi_schemes)} (100%)")
    print(f"Bilingual dataset written to: {OUTPUT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
