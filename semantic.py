"""
semantic.py — Semantic Vector Retrieval Module using Sentence Transformers
===========================================================================
Generates and caches dense semantic embeddings for all schemes using a
multilingual sentence transformer model ('paraphrase-multilingual-MiniLM-L12-v2').

Embeddings are saved to disk (data/embeddings.npy). On subsequent runs, they load
instantly (milliseconds) without recomputation.
"""
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_CACHE_PATH = "data/embeddings.npy"

_model = None

def get_model(name: str = MODEL_NAME) -> SentenceTransformer:
    """Lazy-load the sentence transformer model into memory."""
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model

def make_scheme_semantic_text(scheme: dict) -> str:
    """
    Build a rich bilingual semantic text snippet for embedding.
    Captures both English and Marathi keywords and semantic descriptions.
    """
    parts = [
        scheme.get("scheme_name", ""),
        scheme.get("scheme_name_en", ""),
        scheme.get("category", ""),
        scheme.get("category_en", ""),
        scheme.get("state", ""),
        scheme.get("beneficiary_type", ""),
        scheme.get("description", "")[:400],
        scheme.get("description_en", "")[:400],
        scheme.get("benefits", "")[:250],
        scheme.get("benefits_en", "")[:250],
    ]
    return " | ".join(p.strip() for p in parts if p and p.strip())

def build_or_load_embeddings(schemes: list, cache_path: str = EMBEDDINGS_CACHE_PATH):
    """
    Loads embeddings from cache if available and shape matches.
    Otherwise computes embeddings using sentence-transformers and caches to disk.
    """
    n_schemes = len(schemes)
    if os.path.exists(cache_path):
        try:
            embs = np.load(cache_path)
            if embs.shape[0] == n_schemes:
                return embs
        except Exception:
            pass  # Recompute if corrupted

    print(f"Generating semantic embeddings for {n_schemes} schemes...")
    model = get_model()
    texts = [make_scheme_semantic_text(s) for s in schemes]
    
    # Compute embeddings with normalized vectors for direct dot-product cosine similarity
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embs)
    print(f"Saved semantic embeddings to {cache_path} (shape: {embs.shape})")
    return embs

def semantic_search(query: str, embeddings: np.ndarray, model: SentenceTransformer = None) -> np.ndarray:
    """
    Computes cosine similarity between query and all schemes.
    Returns 1D numpy array of cosine scores [0.0 to 1.0] of length n_schemes.
    """
    if not query or not query.strip() or embeddings is None or len(embeddings) == 0:
        return np.zeros(len(embeddings) if embeddings is not None else 0)
    
    if model is None:
        model = get_model()
        
    q_vec = model.encode([query.strip()], normalize_embeddings=True)[0]
    # Cosine similarity is simply dot product because vectors are normalized
    scores = np.dot(embeddings, q_vec)
    # Clip negative values to 0 for a clean 0.0-1.0 scale
    return np.clip(scores, 0.0, 1.0)

if __name__ == "__main__":
    import json
    with open("data/schemes_bilingual.json", "r", encoding="utf-8") as f:
        schemes = json.load(f)
    print("Pre-computing embeddings...")
    embs = build_or_load_embeddings(schemes)
    print("Testing sample query...")
    sample_queries = ["farmer crop insurance financial aid", "शेतकरी कर्जमाफी आणि अनुदान", "scholarship for higher education girl student"]
    for q in sample_queries:
        sc = semantic_search(q, embs)
        best_idx = np.argmax(sc)
        best_scheme = schemes[best_idx]
        print(f"\nQuery: '{q}' -> Best match ({sc[best_idx]:.3f}):")
        print(f"  MR: {best_scheme['scheme_name']}")
        print(f"  EN: {best_scheme.get('scheme_name_en')}")
