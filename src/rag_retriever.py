import json
import numpy as np
from pathlib import Path
from typing import List, Tuple


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ q


def retrieve(query: str, k: int = 5, index_dir: str = "data") -> List[Tuple[dict, float, str]]:
    from sentence_transformers import SentenceTransformer

    path = Path(index_dir)
    emb_file = path / "song_embeddings.npy"
    idx_file = path / "song_index.json"

    if not emb_file.exists() or not idx_file.exists():
        raise FileNotFoundError(
            "Song index not found. Run `python src/rag_indexer.py` first to build the index."
        )

    embeddings = np.load(emb_file)
    with open(idx_file) as f:
        data = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode(query)
    scores = _cosine_scores(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:k]

    return [(data["songs"][i], float(scores[i]), data["descriptions"][i]) for i in top_indices]
