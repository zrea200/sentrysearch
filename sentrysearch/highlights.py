"""Anomaly-based highlight ranking over the indexed embeddings.

Scores each indexed chunk by how unusual its embedding is and returns the
top-N as candidates for trimming, so users can surface noteworthy moments
without knowing what to search for.
"""

from __future__ import annotations

import numpy as np

from .store import SentryStore


SCORING_METHODS = ("centroid", "knn", "lof")
AGAINST_MODES = ("within", "global")


def _load_index(store: SentryStore) -> tuple[np.ndarray, list[dict]]:
    """Fetch every chunk's embedding and metadata from *store*."""
    data = store.collection.get(include=["embeddings", "metadatas"])
    embeddings = data.get("embeddings")
    metadatas = data.get("metadatas")
    if embeddings is None or len(embeddings) == 0:
        return np.zeros((0, 0)), []
    X = np.asarray(embeddings, dtype=np.float32)
    return X, list(metadatas or [])


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def _cosine_distance_matrix(Xn: np.ndarray) -> np.ndarray:
    """Cosine distance matrix for already-normalized rows. Diagonal is +inf."""
    D = 1.0 - (Xn @ Xn.T)
    np.fill_diagonal(D, np.inf)
    return D


def _score_centroid(Xn: np.ndarray) -> np.ndarray:
    mean = Xn.mean(axis=0)
    mean = mean / max(np.linalg.norm(mean), 1e-12)
    return 1.0 - (Xn @ mean)


def _score_knn(Xn: np.ndarray, k: int) -> np.ndarray:
    n = Xn.shape[0]
    D = _cosine_distance_matrix(Xn)
    k = max(1, min(k, n - 1))
    nearest = np.partition(D, k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def _score_lof(Xn: np.ndarray, k: int) -> np.ndarray:
    n = Xn.shape[0]
    D = _cosine_distance_matrix(Xn)
    k = max(2, min(k, n - 1))
    knn_idx = np.argsort(D, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    k_dist_vec = D[rows, knn_idx[:, k - 1:k]].squeeze(1)  # (n,)
    neigh_dist = D[rows, knn_idx]                         # (n, k)
    reach = np.maximum(k_dist_vec[knn_idx], neigh_dist)   # (n, k)
    lrd = 1.0 / (reach.mean(axis=1) + 1e-12)
    return lrd[knn_idx].mean(axis=1) / (lrd + 1e-12)


def _score(method: str, Xn: np.ndarray, k: int) -> np.ndarray:
    if method == "centroid":
        return _score_centroid(Xn)
    if method == "knn":
        return _score_knn(Xn, k)
    if method == "lof":
        return _score_lof(Xn, k)
    raise ValueError(f"unknown method: {method}")


def _exclude_baseline_mask(Xn: np.ndarray) -> np.ndarray:
    """Keep the half of points farthest from the index centroid."""
    n = Xn.shape[0]
    if n < 4:
        return np.ones(n, dtype=bool)
    mean = Xn.mean(axis=0)
    mean = mean / max(np.linalg.norm(mean), 1e-12)
    dist = 1.0 - (Xn @ mean)
    cutoff = np.median(dist)
    return dist >= cutoff


def _dedupe_indices(
    ranked: np.ndarray,
    Xn: np.ndarray,
    threshold: float,
    limit: int,
) -> list[int]:
    """Greedy MMR-style dedupe: drop entries too similar to a higher-ranked pick."""
    kept: list[int] = []
    for idx in ranked:
        i = int(idx)
        if not kept:
            kept.append(i)
        else:
            sims = Xn[kept] @ Xn[i]
            if float(sims.max()) <= threshold:
                kept.append(i)
        if len(kept) >= limit:
            break
    return kept


def rank_highlights(
    store: SentryStore,
    *,
    count: int,
    method: str = "knn",
    neighbors: int = 10,
    dedupe_threshold: float = 0.9,
    exclude_baseline: bool = False,
    against_embedding: np.ndarray | None = None,
    against_mode: str = "within",
    against_pool: int = 50,
) -> list[dict]:
    """Return up to *count* result dicts (source_file/start_time/end_time/score).

    When *against_embedding* is given:
      - "within": restrict scoring to the top-*against_pool* matches of the query.
      - "global": score over the full index, then weight by query similarity.
    """
    if method not in SCORING_METHODS:
        raise ValueError(f"method must be one of {SCORING_METHODS}")
    if against_mode not in AGAINST_MODES:
        raise ValueError(f"against-mode must be one of {AGAINST_MODES}")

    X, metas = _load_index(store)
    n = X.shape[0]
    if n == 0:
        return []

    Xn = _normalize(X)
    candidate_mask = np.ones(n, dtype=bool)

    query_sim: np.ndarray | None = None
    if against_embedding is not None:
        q = against_embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        query_sim = Xn @ q
        if against_mode == "within":
            pool = min(max(against_pool, count), n)
            top_idx = np.argpartition(-query_sim, pool - 1)[:pool]
            candidate_mask[:] = False
            candidate_mask[top_idx] = True

    if exclude_baseline:
        candidate_mask &= _exclude_baseline_mask(Xn)

    cand_idx = np.where(candidate_mask)[0]
    if cand_idx.size == 0:
        return []
    if cand_idx.size < n:
        Xn_sub = Xn[cand_idx]
    else:
        Xn_sub = Xn

    if Xn_sub.shape[0] < 2:
        scores_sub = np.zeros(Xn_sub.shape[0])
    else:
        scores_sub = _score(method, Xn_sub, neighbors)

    if query_sim is not None and against_mode == "global":
        sim_sub = np.clip(query_sim[cand_idx], 0.0, None)
        s = scores_sub
        s_norm = (s - s.min()) / max(float(s.max() - s.min()), 1e-12)
        scores_sub = s_norm * sim_sub

    order = np.argsort(-scores_sub)
    ranked_global = cand_idx[order]
    kept = _dedupe_indices(ranked_global, Xn, dedupe_threshold, count)

    results = []
    for i in kept:
        m = metas[i]
        # Recover this point's score from scores_sub via position in cand_idx
        local = int(np.where(cand_idx == i)[0][0])
        results.append({
            "source_file": m["source_file"],
            "start_time": float(m["start_time"]),
            "end_time": float(m["end_time"]),
            "similarity_score": float(scores_sub[local]),
        })
    return results
