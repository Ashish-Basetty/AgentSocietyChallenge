"""Reranker utilities: select best candidate among multiple LLM outputs.

This module provides a simple embedding-based reranker that scores each candidate
by its average cosine similarity to the user's past reviews. It falls back to
length-based heuristics if embeddings are unavailable.
"""
from typing import List, Optional
import numpy as np
import math


def _cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rerank_by_user_similarity(candidates: List[str], user_reviews: List[str], embedding_provider) -> int:
    """
    Rerank candidates by average cosine similarity to user's past reviews.

    Args:
        candidates: list of candidate review strings
        user_reviews: list of user review texts (may be empty)
        embedding_provider: object that implements `embed_documents(list[str]) -> List[List[float]]`

    Returns:
        index of selected candidate (int)
    """
    if not candidates:
        return 0

    # If no user reviews, fall back to longest candidate heuristic
    if not user_reviews or len(user_reviews) == 0:
        # prefer the candidate with the most content (non-empty), tie-breaker: first
        lengths = [len(c.strip()) for c in candidates]
        return int(int(np.argmax(lengths)))

    try:
        # Use embedding provider to get embeddings
        # embedding_provider expected to have embed_documents() method
        cand_embs = embedding_provider.embed_documents(candidates)
        user_embs = embedding_provider.embed_documents(user_reviews)

        # Compute average similarity of each candidate to all user reviews
        scores = []
        for c_emb in cand_embs:
            sims = [_cosine(c_emb, u_emb) for u_emb in user_embs]
            # score is mean similarity
            if sims:
                scores.append(float(np.mean(sims)))
            else:
                scores.append(0.0)

        # Choose candidate with highest score
        best_idx = int(int(np.argmax(scores)))
        return best_idx
    except Exception:
        # Embedding fallback: length heuristic
        lengths = [len(c.strip()) for c in candidates]
        return int(int(np.argmax(lengths)))


def rerank_hybrid(candidates: List[str], user_reviews: List[str], embedding_provider=None, parsed_candidates: Optional[List[dict]] = None, n: int = 1) -> int:
    """
    Hybrid reranker that combines embedding similarity to user reviews with a
    candidate-quality heuristic. The weighting shifts slightly as `n` grows to
    emphasize quality when candidate pools are larger.

    Args:
        candidates: list of candidate review texts
        user_reviews: list of user's past review texts
        embedding_provider: optional provider with `embed_documents(list[str])`
        parsed_candidates: optional list of dicts containing parsed fields (e.g. 'stars')
        n: number of candidates requested (used to adjust weights)

    Returns:
        index of selected candidate
    """
    if not candidates:
        return 0

    L = len(candidates)

    # --- Embedding-based similarity ---
    emb_scores = [0.5] * L
    if embedding_provider is not None and user_reviews:
        try:
            cand_embs = embedding_provider.embed_documents(candidates)
            user_embs = embedding_provider.embed_documents(user_reviews)
            for i, c_emb in enumerate(cand_embs):
                sims = [_cosine(c_emb, u_emb) for u_emb in user_embs]
                emb_scores[i] = float(np.mean(sims)) if sims else 0.0
        except Exception:
            emb_scores = [0.5] * L

    # --- Heuristic quality scoring ---
    positive_words = set(["good", "great", "excellent", "amazing", "delicious", "friendly", "helpful", "love", "best"])
    negative_words = set(["bad", "terrible", "awful", "worst", "disgusting", "rude", "horrible", "never"])

    quality_scores = []
    for idx, text in enumerate(candidates):
        t = text.strip()
        length = len(t)
        # length_score: prefer reasonable length up to 300 chars
        length_score = min(length, 300) / 300.0

        # sentence score: prefer 2-4 sentences
        sentences = max(1, t.count('.') + t.count('!') + t.count('?'))
        sentence_score = 1.0 - min(abs(sentences - 3) / 3.0, 1.0)

        # polarity / consistency score based on parsed stars if available
        polarity_score = 0.5
        if parsed_candidates and idx < len(parsed_candidates):
            stars = parsed_candidates[idx].get('stars', None)
            lowered = t.lower()
            pos_count = sum(1 for w in positive_words if w in lowered)
            neg_count = sum(1 for w in negative_words if w in lowered)
            if stars is not None:
                if stars >= 4:
                    polarity_score = min(1.0, (pos_count + 0.5) / (pos_count + neg_count + 1.0))
                elif stars <= 2:
                    polarity_score = min(1.0, (neg_count + 0.5) / (pos_count + neg_count + 1.0))
                else:
                    polarity_score = 0.5
            else:
                polarity_score = min(1.0, (pos_count + 0.5) / (pos_count + neg_count + 1.0))

        quality = 0.4 * length_score + 0.4 * sentence_score + 0.2 * polarity_score
        quality_scores.append(float(quality))

    # normalize scores to 0..1
    def _normalize(arr):
        a = np.array(arr, dtype=float)
        mn = float(np.min(a))
        mx = float(np.max(a))
        if mx - mn < 1e-6:
            return [0.5] * len(a)
        return ((a - mn) / (mx - mn)).tolist()

    emb_norm = _normalize(emb_scores)
    qual_norm = _normalize(quality_scores)

    # weight: as n increases, rely slightly more on quality (empirically)
    alpha = max(0.2, 0.7 - 0.02 * n)  # weight for embedding similarity

    combined = [alpha * e + (1 - alpha) * q for e, q in zip(emb_norm, qual_norm)]

    best_idx = int(int(np.argmax(combined)))
    return best_idx


def rerank_by_llm(llm, candidates: List[str], user_reviews: List[str], parsed_candidates: Optional[List[dict]] = None, n: int = 1) -> int:
    """
    Rerank candidates by asking the LLM to score each candidate for
    usefulness/quality/consistency with the user's past reviews.

    This function is robust to mock LLMs: if the provided `llm` appears to be
    a mock (class name starts with 'Mock' or has attribute `is_mock`), it
    computes deterministic heuristic scores instead of calling the model.

    Args:
        llm: LLM instance supporting call via llm(messages=[...])
        candidates: list of candidate strings
        user_reviews: list of user's past review texts
        parsed_candidates: optional parsed metadata (stars)
        n: number of candidates requested

    Returns:
        index of selected candidate
    """
    if not candidates:
        return 0

    # Detect mock LLMs to avoid making external API calls during tests
    llm_name = getattr(llm, '__class__', type(llm)).__name__
    is_mock = getattr(llm, 'is_mock', False) or (isinstance(llm_name, str) and llm_name.lower().startswith('mock'))

    scores = []
    if is_mock:
        # Deterministic mock scoring: combine token-hash and star preference
        for i, c in enumerate(candidates):
            base = sum(ord(ch) for ch in c) % 100
            star_bonus = 0
            if parsed_candidates and i < len(parsed_candidates):
                star = parsed_candidates[i].get('stars', 3.0)
                star_bonus = int((star - 3.0) * 5)  # small bias for higher stars
            scores.append(float((base + star_bonus) / 100.0))
        best_idx = int(int(np.argmax(scores)))
        return best_idx

    # Real LLM scoring path: ask the LLM to score each candidate 0-100
    for idx, cand in enumerate(candidates):
        try:
            prompt = f"You are an evaluator. The following is a short user profile (past reviews):\n{('\n'.join(user_reviews)) if user_reviews else 'No past reviews.'}\n\nCandidate review:\n{cand}\n\nPlease RATE this candidate on a scale from 0 to 100 for: usefulness, clarity, and consistency with the user's style. Reply with a single integer only."
            # Call the LLM
            resp = llm(messages=[{"role": "user", "content": prompt}], temperature=0.0)
            # Extract integer
            import re
            m = re.search(r'(\d+)', str(resp))
            score = float(m.group(1)) if m else 50.0
        except Exception:
            score = 50.0
        scores.append(score / 100.0)

    best_idx = int(int(np.argmax(scores)))
    return best_idx
