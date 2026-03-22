from __future__ import annotations

import re
from functools import lru_cache

from app.config import RERANK_BACKEND, RERANK_MODEL
from app.schemas import PaperRecord


def _extract_terms(text: str) -> list[str]:
    terms: list[str] = []
    for tok in re.split(r"[^a-zA-Z0-9+.-]+", (text or "").lower()):
        if len(tok) < 3:
            continue
        terms.append(tok)
    return terms


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _lexical_score(query: str, doc: str) -> float:
    q_terms = _extract_terms(query)
    if not q_terms:
        return 0.0

    d_terms = set(_extract_terms(doc))
    if not d_terms:
        return 0.0

    hit = sum(1 for t in q_terms if t in d_terms)
    coverage = hit / max(1, min(12, len(set(q_terms))))

    phrase_bonus = 0.0
    doc_low = (doc or "").lower()
    for phrase in [
        "retrieval augmented generation",
        "cross encoder",
        "cross-encoder",
        "markov decision process",
        "task planning",
        "workflow scheduling",
        "large language model",
    ]:
        if phrase in query.lower() and phrase in doc_low:
            phrase_bonus += 0.08
    return coverage + phrase_bonus


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _cross_encoder_scores(query: str, docs: list[str], model_name: str) -> tuple[list[float] | None, str]:
    try:
        model = _load_cross_encoder(model_name)
    except Exception as exc:
        return None, f"cross-encoder load failed: {exc}"

    try:
        pairs = [(query, d) for d in docs]
        raw_scores = model.predict(pairs)
        values = [float(v) for v in raw_scores]
        return _minmax(values), ""
    except Exception as exc:
        return None, f"cross-encoder inference failed: {exc}"


def _compose_query(question: str, sub_queries: list[str], locked_concepts: list[str]) -> str:
    blocks = [question.strip()]
    blocks.extend([q.strip() for q in sub_queries if q and q.strip()])
    blocks.extend([c.strip() for c in locked_concepts if c and c.strip()])
    uniq: list[str] = []
    for b in blocks:
        b_low = b.lower()
        if b_low not in [x.lower() for x in uniq]:
            uniq.append(b)
    return " ; ".join(uniq)


def rerank_papers(
    question: str,
    papers: list[PaperRecord],
    sub_queries: list[str],
    locked_concepts: list[str] | None = None,
    *,
    top_k: int = 30,
    backend: str | None = None,
    model_name: str | None = None,
) -> tuple[list[PaperRecord], dict]:
    top_k = max(1, min(int(top_k or 30), 30))
    locked = locked_concepts or []
    if not papers:
        return [], {"backend": "none", "model": "", "fallback_reason": "empty candidates"}

    query = _compose_query(question, sub_queries, locked)
    docs = [
        " ".join(
            [
                p.title or "",
                p.abstract or "",
                p.venue or "",
                " ".join(p.authors or []),
            ]
        ).strip()
        for p in papers
    ]

    lexical_raw = [_lexical_score(query, d) for d in docs]
    lexical_scores = _minmax(lexical_raw)

    chosen_backend = (backend or RERANK_BACKEND).strip().lower()
    if chosen_backend not in {"auto", "cross_encoder", "cross-encoder", "lexical"}:
        chosen_backend = "auto"
    chosen_model = (model_name or RERANK_MODEL).strip()

    cross_scores: list[float] | None = None
    fallback_reason = ""
    if chosen_backend in {"auto", "cross_encoder", "cross-encoder"}:
        cross_scores, fallback_reason = _cross_encoder_scores(query, docs, chosen_model)

    scored_rows: list[tuple[float, float, int, int, PaperRecord]] = []
    for i, paper in enumerate(papers):
        lex = lexical_scores[i] if i < len(lexical_scores) else 0.0
        if cross_scores is not None and i < len(cross_scores):
            final = 0.75 * cross_scores[i] + 0.25 * lex
            backend_used = "cross_encoder"
        else:
            final = lex
            backend_used = "lexical"

        paper.rerank_score = round(final * 100.0, 4)
        scored_rows.append(
            (
                paper.rerank_score,
                paper.query_match_score,
                paper.citation_count,
                paper.year or 0,
                paper,
            )
        )

    scored_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    reranked = [x[-1] for x in scored_rows[:top_k]]
    info = {
        "backend": backend_used,
        "model": chosen_model if backend_used == "cross_encoder" else "",
        "fallback_reason": fallback_reason if backend_used != "cross_encoder" else "",
        "input_candidates": len(papers),
        "output_candidates": len(reranked),
    }
    return reranked, info
