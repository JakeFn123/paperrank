from __future__ import annotations

import hashlib
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
import math

import httpx

from app.config import MCP_SERVER_PATH, SEARCH_RECALL_POOL, SEARCH_TOP_K
from app.rerank import rerank_papers
from app.schemas import PaperRecord


def _as_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_paper_row(row: dict) -> dict:
    normalized = dict(row)
    normalized["paper_id"] = _as_str(normalized.get("paper_id"))
    normalized["title"] = _as_str(normalized.get("title"))
    normalized["abstract"] = _as_str(normalized.get("abstract"))
    normalized["venue"] = _as_str(normalized.get("venue"))
    normalized["source"] = _as_str(normalized.get("source"))
    normalized["paper_url"] = _as_str(normalized.get("paper_url"))
    normalized["pdf_url"] = _as_str(normalized.get("pdf_url"))

    try:
        normalized["citation_count"] = int(normalized.get("citation_count") or 0)
    except Exception:
        normalized["citation_count"] = 0

    year_val = normalized.get("year")
    try:
        normalized["year"] = int(year_val) if year_val not in (None, "", "null") else None
    except Exception:
        normalized["year"] = None

    authors = normalized.get("authors")
    if isinstance(authors, list):
        normalized["authors"] = [_as_str(a) for a in authors if a is not None]
    elif authors is None:
        normalized["authors"] = []
    else:
        normalized["authors"] = [_as_str(authors)]
    return normalized


def _stable_id(title: str, year: int | None, source: str) -> str:
    key = f"{_as_str(title).lower().strip()}::{year or 0}::{_as_str(source)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _extract_terms(text: str) -> list[str]:
    terms = []
    for tok in re.split(r"[^a-zA-Z0-9+.-]+", (text or "").lower()):
        if len(tok) < 3:
            continue
        terms.append(tok)
    uniq = []
    for t in terms:
        if t not in uniq:
            uniq.append(t)
    return uniq


def _simplify_query(query: str, max_terms: int = 10) -> str:
    terms = _extract_terms(query)
    if not terms:
        return ""
    return " ".join(terms[:max_terms]).strip()


def _query_anchor_terms(question: str, sub_queries: list[str], locked_concepts: list[str]) -> set[str]:
    anchors = set(_extract_terms(question))
    for q in sub_queries:
        anchors.update(_extract_terms(q))
    for c in locked_concepts:
        anchors.update(_extract_terms(c))
    return anchors


def _required_concept_groups(question: str, locked_concepts: list[str]) -> list[tuple[str, set[str], str]]:
    q = (question or "").lower()
    groups: list[tuple[str, set[str], str]] = []

    if any(k in q for k in ["rag", "检索增强", "retrieval augmented generation"]):
        groups.append(
            (
                "RAG / Retrieval-Augmented Generation",
                {"rag", "retrieval", "augmented", "generation", "retrieval-augmented"},
                "any",
            )
        )
    if any(k in q for k in ["rerank", "reranker", "重排", "重排序", "cross-encoder"]):
        groups.append(
            (
                "Reranking / Cross-Encoder",
                {"rerank", "reranker", "re-ranking", "cross-encoder", "crossencoder"},
                "any",
            )
        )
    if any(k in q for k in ["企业", "knowledge base", "问答", "question answering", "enterprise"]):
        groups.append(
            (
                "Enterprise QA / Knowledge Base",
                {"question answering", "qa", "knowledge base", "enterprise"},
                "any",
            )
        )
    if any(k in q for k in ["稳定", "鲁棒", "reliability", "robustness", "hallucination"]):
        groups.append(
            (
                "Robustness / Reliability / Faithfulness",
                {"robustness", "reliability", "hallucination", "faithfulness", "stability"},
                "any",
            )
        )

    for concept in locked_concepts:
        c = (concept or "").strip().lower()
        if not c:
            continue
        label = f"Locked: {concept}"
        groups.append((label, {c}, "phrase"))

    return groups


def _matched_concept_labels(row: dict, groups: list[tuple[str, set[str], str]]) -> list[str]:
    if not groups:
        return []
    text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
    labels = []
    for label, terms, mode in groups:
        if mode == "all":
            if all(term in text for term in terms):
                labels.append(label)
        elif mode == "phrase":
            phrase = next(iter(terms), "").strip()
            if phrase and phrase in text:
                labels.append(label)
        elif any(term in text for term in terms):
            labels.append(label)
    return labels


def _concept_phrase_match(text: str, concept: str) -> bool:
    concept = (concept or "").strip().lower()
    if not concept:
        return False
    if " " in concept:
        return concept in text
    # single-token concept, use token boundary matching
    return re.search(rf"\b{re.escape(concept)}\b", text) is not None


def _locked_concept_match_count(row: dict, locked_concepts: list[str]) -> int:
    if not locked_concepts:
        return 0
    text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
    count = 0
    for concept in locked_concepts:
        concept = (concept or "").strip()
        if not concept:
            continue
        c_low = concept.lower()
        if _concept_phrase_match(text, c_low):
            count += 1
            continue

        # Fallback for long concepts: accept partial token coverage.
        tokens = [t for t in _extract_terms(c_low) if t]
        if len(tokens) >= 3:
            hit = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", text))
            if hit >= max(2, len(tokens) - 1):
                count += 1
        elif len(tokens) == 2:
            if any(re.search(rf"\b{re.escape(t)}\b", text) for t in tokens):
                count += 1
    return count


def _row_matches_locked_concepts(row: dict, locked_concepts: list[str]) -> bool:
    if not locked_concepts:
        return True
    matched = _locked_concept_match_count(row, locked_concepts)
    # Adaptive threshold to avoid zero-recall:
    # 1 concept -> need 1 hit
    # 2-3 concepts -> need at least 1 hit
    # >=4 concepts -> need at least 2 hits
    if len(locked_concepts) == 1:
        required = 1
    elif len(locked_concepts) <= 3:
        required = 1
    else:
        required = 2
    return matched >= required


def _relevance_score(row: dict, anchors: set[str]) -> float:
    if not anchors:
        return 0.0
    text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
    if not text.strip():
        return 0.0
    tokens = set(_extract_terms(text))
    if not tokens:
        return 0.0
    inter = len(tokens & anchors)
    coverage = inter / max(1, min(10, len(anchors)))

    # Reward exact technical phrase hits for high-precision queries.
    phrase_bonus = 0.0
    lower_text = text
    for phrase in [
        "retrieval augmented generation",
        "reranker",
        "reranking",
        "question answering",
        "hallucination",
        "robustness",
        "reliability",
    ]:
        if phrase in lower_text and phrase in " ".join(anchors):
            phrase_bonus += 0.1
    return round(coverage + phrase_bonus, 4)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ToolServerClient:
    def __init__(self) -> None:
        self.port = _pick_free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "ToolServerClient":
        self.proc = subprocess.Popen(
            [sys.executable, MCP_SERVER_PATH, "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_ready()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _wait_ready(self, timeout: float = 8.0) -> None:
        start = time.time()
        with httpx.Client(timeout=2.0) as client:
            while time.time() - start < timeout:
                try:
                    r = client.get(f"{self.base}/health")
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                time.sleep(0.1)
        raise RuntimeError("Tool server failed to start")

    def call_tool(self, name: str, arguments: dict) -> list[dict] | dict:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base}/tools/call", json={"name": name, "arguments": arguments})
            payload = response.json()
            if response.status_code >= 400:
                raise RuntimeError(payload.get("error", f"tool call failed with status {response.status_code}"))
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload.get("result", [])



def dedupe_papers(
    rows: list[dict],
    question: str,
    sub_queries: list[str],
    locked_concepts: list[str] | None = None,
    top_k: int = SEARCH_TOP_K,
    enforce_ui_cap: bool = True,
) -> list[PaperRecord]:
    if enforce_ui_cap:
        top_k = max(1, min(int(top_k or SEARCH_TOP_K), 30))
    else:
        top_k = max(1, min(int(top_k or SEARCH_TOP_K), 300))
    locked_concepts = locked_concepts or []
    bucket: dict[str, dict] = {}
    for raw in rows:
        row = _normalize_paper_row(raw)
        pid = row.get("paper_id") or _stable_id(row.get("title", ""), row.get("year"), row.get("source", ""))
        existing = bucket.get(pid)
        if existing is None:
            bucket[pid] = row
            continue

        if len((row.get("abstract") or "")) > len((existing.get("abstract") or "")):
            bucket[pid]["abstract"] = row.get("abstract")
        if row.get("pdf_url") and not existing.get("pdf_url"):
            bucket[pid]["pdf_url"] = row.get("pdf_url")
        if row.get("paper_url") and not existing.get("paper_url"):
            bucket[pid]["paper_url"] = row.get("paper_url")

    anchors = _query_anchor_terms(question, sub_queries, locked_concepts)
    concept_groups = _required_concept_groups(question, locked_concepts)
    with_scores = []
    fallback_rows = []
    for pid, row in bucket.items():
        lock_match_count = _locked_concept_match_count(row, locked_concepts)
        passes_locked = _row_matches_locked_concepts(row, locked_concepts)
        matched_concepts = _matched_concept_labels(row, concept_groups)
        concept_hits = len(matched_concepts)
        paper = PaperRecord.model_validate({**row, "paper_id": row.get("paper_id") or pid})
        rel = _relevance_score(row, anchors)
        query_match_score = round(concept_hits * 1.0 + rel + lock_match_count * 1.5, 4)
        paper.query_match_score = query_match_score
        paper.concept_hit_count = concept_hits
        paper.matched_concepts = matched_concepts
        if passes_locked or not locked_concepts:
            with_scores.append((concept_hits, rel, lock_match_count, paper))
        else:
            fallback_rows.append((concept_hits, rel, lock_match_count, paper))

    # If strict/soft locked filtering removed everything, fallback to best partial matches.
    if not with_scores and fallback_rows:
        with_scores = fallback_rows

    # Rank by concept-group hits first, then lexical relevance, then citation/year.
    with_scores.sort(
        key=lambda x: (x[2], x[0], x[1], x[3].citation_count, x[3].year or 0),
        reverse=True,
    )
    papers = [p for _, _, _, p in with_scores]

    # Hard filter: only enforce for small top_k modes.
    # For larger candidate budgets (e.g., 30), keep tail candidates to preserve recall.
    if top_k <= 10:
        positive = [p for ch, _, _, p in with_scores if ch > 0]
        if len(positive) >= min(top_k, 3):
            papers = positive

    # If every row has zero relevance, fall back to citation/year.
    if with_scores and all(rel <= 0 for _, rel, _, _ in with_scores):
        papers.sort(key=lambda x: (x.citation_count, x.year or 0), reverse=True)
    return papers[:top_k]


async def search_via_mcp(
    question: str,
    sub_queries: list[str],
    source: str = "all",
    per_query_limit: int = 6,
    locked_concepts: list[str] | None = None,
    max_papers: int = SEARCH_TOP_K,
) -> tuple[list[PaperRecord], list[dict]]:
    logs: list[dict] = []
    collected: list[dict] = []
    recall_target = max(max_papers, min(int(SEARCH_RECALL_POOL or 100), 200))
    source_multiplier = 2 if str(source).lower().strip() == "all" else 1
    subquery_count = max(1, len(sub_queries))
    dynamic_limit = math.ceil(recall_target / (subquery_count * source_multiplier))
    effective_limit = max(1, min(max(per_query_limit, dynamic_limit), 20))

    with ToolServerClient() as server:
        for q in sub_queries:
            try:
                rows = server.call_tool(
                    "search_papers",
                    {
                        "query": q,
                        "limit": effective_limit,
                        "source": source,
                        "locked_concepts": locked_concepts or [],
                    },
                )
                if not isinstance(rows, list):
                    rows = []

                fallback_query = ""
                fallback_hits = 0
                if not rows:
                    fallback_query = _simplify_query(q, max_terms=10)
                    if fallback_query and fallback_query != q:
                        try:
                            retry_rows = server.call_tool(
                                "search_papers",
                                {
                                    "query": fallback_query,
                                    "limit": effective_limit,
                                    "source": source,
                                    "locked_concepts": locked_concepts or [],
                                },
                            )
                            if isinstance(retry_rows, list) and retry_rows:
                                rows = retry_rows
                                fallback_hits = len(retry_rows)
                        except Exception:
                            pass
                logs.append(
                    {
                        "query": q,
                        "hits": len(rows),
                        "source": source,
                        "status": "ok",
                        "requested_limit": per_query_limit,
                        "effective_limit": effective_limit,
                        "fallback_query": fallback_query,
                        "fallback_hits": fallback_hits,
                    }
                )
            except Exception as exc:
                rows = []
                logs.append(
                    {
                        "query": q,
                        "hits": 0,
                        "source": source,
                        "status": "error",
                        "error": str(exc),
                        "requested_limit": per_query_limit,
                        "effective_limit": effective_limit,
                    }
                )
            collected.extend(rows)

    recall_pool = dedupe_papers(
        collected,
        question=question,
        sub_queries=sub_queries,
        locked_concepts=locked_concepts or [],
        top_k=recall_target,
        enforce_ui_cap=False,
    )
    papers, rerank_info = rerank_papers(
        question=question,
        papers=recall_pool,
        sub_queries=sub_queries,
        locked_concepts=locked_concepts or [],
        top_k=max_papers,
    )
    logs.append(
        {
            "query": "__postprocess__",
            "hits": len(papers),
            "source": "local",
            "status": "ok",
            "recall_target": recall_target,
            "recall_after_dedupe": len(recall_pool),
            "rerank_backend": rerank_info.get("backend", ""),
            "rerank_model": rerank_info.get("model", ""),
            "rerank_fallback": rerank_info.get("fallback_reason", ""),
        }
    )
    return papers, logs


def compute_complementarity(papers: list[PaperRecord]) -> dict[str, float]:
    token_sets = {}
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        tokens = {tok for tok in text.split() if len(tok) > 3}
        token_sets[p.paper_id] = tokens

    scores: dict[str, float] = defaultdict(float)
    for i, pa in enumerate(papers):
        set_a = token_sets[pa.paper_id]
        if not set_a:
            scores[pa.paper_id] = 50.0
            continue
        similarities = []
        for j, pb in enumerate(papers):
            if i == j:
                continue
            set_b = token_sets[pb.paper_id]
            if not set_b:
                continue
            inter = len(set_a & set_b)
            union = len(set_a | set_b) or 1
            similarities.append(inter / union)
        mean_sim = sum(similarities) / len(similarities) if similarities else 0.0
        scores[pa.paper_id] = round((1 - mean_sim) * 100, 2)
    return scores


def compute_quality_signal(paper: PaperRecord) -> float:
    citation_signal = min(100.0, (paper.citation_count / 300.0) * 100.0)
    venue_signal = 70.0 if paper.venue and paper.venue.lower() != "arxiv" else 50.0
    author_signal = min(100.0, 40.0 + len(paper.authors) * 8.0)
    return round(citation_signal * 0.45 + venue_signal * 0.35 + author_signal * 0.20, 2)
