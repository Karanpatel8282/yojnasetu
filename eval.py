"""
eval.py  —  Evaluation Benchmark (Precision@K / Recall@K)
=========================================================
Benchmarks the hybrid retrieval engine on a curated gold set of
both Marathi and English queries.
"""
import sys
import numpy as np
from engine import load_schemes, build_index, rank

sys.stdout.reconfigure(encoding="utf-8")

NEUTRAL = dict(age=25, gender="all", state="all", income=0,
               occupation="all", residence="both", bpl=False,
               disability=False, caste="all")

GOLD = {
    # Marathi Queries
    "शिष्यवृत्ती": ["gpcs", "cmss", "sdsspd", "caicwacs"],
    "शेतकरी कर्ज": ["cmkry", "tlgntdnt", "phms", "mmvyjsy"],
    "महिला कर्ज": ["mssg", "msymc", "mkyh", "msybc"],
    "पेन्शन ज्येष्ठ": ["psfa", "gspsg", "pbftcw", "wdwps"],
    "गृहनिर्माण": ["v-vhcs", "cmbocwhrs", "cmhap", "dbjrlcwscs"],
    # English Queries
    "scholarship student": ["gpcs", "cmss", "sdsspd", "caicwacs"],
    "farmer loan": ["cmkry", "tlgntdnt", "phms", "mmvyjsy"],
}


def precision_recall(ranked_ids, gold_ids, k):
    gold = set(gold_ids)
    top = set(ranked_ids[:k])
    inter = top & gold
    p = len(inter) / k
    r = len(inter) / len(gold) if gold else 0.0
    return p, r


def main():
    schemes = load_schemes()
    index_bundle, legacy_matrix = build_index(schemes)

    print("=" * 74)
    print("HYBRID RETRIEVAL BENCHMARK  (%d queries)" % len(GOLD))
    print("=" * 74)

    rows = []
    for query, gold in GOLD.items():
        res = rank(schemes, index_bundle, legacy_matrix, query, NEUTRAL, top_k=10)
        blended_ids = [r["scheme"]["scheme_id"] for r in res]

        p5, r5 = precision_recall(blended_ids, gold, 5)
        p10, r10 = precision_recall(blended_ids, gold, 10)

        rows.append((query, p5, r5, p10, r10))
        lang = "EN" if any(c.isascii() and c.isalpha() for c in query) else "MR"
        print(f"\nQ [{lang}]: {query}")
        print(f"   Hybrid: P@5={p5:.2f}, R@5={r5:.2f} | P@10={p10:.2f}, R@10={r10:.2f}")

    print("\n" + "-" * 74)
    print("BENCHMARK SUMMARY")
    print(f"Mean P@5 : {np.mean([r[1] for r in rows]):.2f}")
    print(f"Mean R@5 : {np.mean([r[2] for r in rows]):.2f}")
    print(f"Mean P@10: {np.mean([r[3] for r in rows]):.2f}")
    print(f"Mean R@10: {np.mean([r[4] for r in rows]):.2f}")
    print("=" * 74)


if __name__ == "__main__":
    main()