from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import feedparser
import httpx
from dotenv import load_dotenv

# Ensure project root is importable when this file is launched directly.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import SEMANTIC_SCHOLAR_API_KEY

load_dotenv()

_ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
_SEMANTIC_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_ARXIV_LOCK = threading.Lock()
_LAST_ARXIV_REQUEST: datetime | None = None


def _arxiv_rate_limit() -> None:
    global _LAST_ARXIV_REQUEST
    with _ARXIV_LOCK:
        if _LAST_ARXIV_REQUEST is not None:
            elapsed = datetime.now() - _LAST_ARXIV_REQUEST
            if elapsed < timedelta(seconds=3):
                delay = 3 - elapsed.total_seconds()
                if delay > 0:
                    import time

                    time.sleep(delay)
        _LAST_ARXIV_REQUEST = datetime.now()


def _build_arxiv_search_query(raw_query: str, locked_concepts: list[str] | None = None) -> str:
    q = (raw_query or "").lower()
    clauses: list[str] = []

    if "retrieval augmented generation" in q or " rag " in f" {q} " or q.startswith("rag "):
        clauses.append('all:"retrieval augmented generation"')
    elif "retrieval" in q and "generation" in q:
        clauses.append("(all:retrieval AND all:generation)")

    if any(k in q for k in ["rerank", "reranker", "cross-encoder", "cross encoder"]):
        clauses.append('(all:rerank OR all:reranker OR all:"cross encoder")')

    if any(k in q for k in ["question answering", "qa", "knowledge base", "enterprise"]):
        clauses.append('(all:"question answering" OR all:"knowledge base" OR all:enterprise)')

    if any(k in q for k in ["robustness", "reliability", "hallucination", "faithfulness", "stability"]):
        clauses.append('(all:robustness OR all:reliability OR all:hallucination OR all:faithfulness)')

    # Keep one broad fallback term to avoid over-constraining.
    if "benchmark" in q:
        clauses.append("all:benchmark")
    elif "ablation" in q:
        clauses.append("all:ablation")

    locked_clauses = []
    for concept in (locked_concepts or [])[:3]:
        c = str(concept or "").strip().lower()
        if not c:
            continue
        if " " in c:
            locked_clauses.append(f'all:"{c}"')
        else:
            locked_clauses.append(f"all:{c}")
    if locked_clauses:
        # Use OR for locked concepts at retrieval stage to keep recall,
        # then apply concept-aware ranking/filtering in app/tooling.py.
        clauses.append("(" + " OR ".join(locked_clauses) + ")")

    if not clauses:
        return raw_query
    return " AND ".join(clauses)


async def _search_semantic_scholar(query: str, limit: int, locked_concepts: list[str] | None = None) -> list[dict[str, Any]]:
    final_query = (query or "").strip()
    for concept in (locked_concepts or [])[:3]:
        c = str(concept or "").strip()
        if not c:
            continue
        if c.lower() not in final_query.lower():
            final_query = f"{final_query} {c}".strip()

    fields = ",".join(
        [
            "paperId",
            "title",
            "abstract",
            "year",
            "venue",
            "citationCount",
            "authors",
            "url",
            "openAccessPdf",
        ]
    )
    headers: dict[str, str] = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    data: list[dict[str, Any]] = []
    max_attempts = 3
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(max_attempts):
            try:
                response = await client.get(
                    _SEMANTIC_SEARCH_URL,
                    params={"query": final_query, "limit": limit, "fields": fields},
                    headers=headers,
                )

                # Gracefully handle public API rate limiting.
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            sleep_sec = max(1.0, float(retry_after))
                        except Exception:
                            sleep_sec = 1.5 * (attempt + 1)
                    else:
                        sleep_sec = 1.5 * (attempt + 1)
                    await asyncio.sleep(sleep_sec)
                    continue

                response.raise_for_status()
                data = response.json().get("data", [])
                break
            except httpx.HTTPStatusError:
                # Non-429 HTTP failures fall back to empty list.
                if attempt == max_attempts - 1:
                    return []
                await asyncio.sleep(1.0 * (attempt + 1))
            except Exception:
                if attempt == max_attempts - 1:
                    return []
                await asyncio.sleep(1.0 * (attempt + 1))

    papers: list[dict[str, Any]] = []
    for item in data:
        papers.append(
            {
                "paper_id": item.get("paperId") or "",
                "title": item.get("title") or "",
                "abstract": item.get("abstract") or "",
                "year": item.get("year"),
                "venue": item.get("venue") or "",
                "citation_count": int(item.get("citationCount") or 0),
                "authors": [a.get("name") or "" for a in item.get("authors", [])],
                "source": "semantic_scholar",
                "paper_url": item.get("url") or "",
                "pdf_url": (item.get("openAccessPdf") or {}).get("url", ""),
            }
        )
    return papers


async def _search_arxiv(query: str, limit: int, locked_concepts: list[str] | None = None) -> list[dict[str, Any]]:
    _arxiv_rate_limit()
    search_query = _build_arxiv_search_query(query, locked_concepts=locked_concepts)

    params = {
        "search_query": search_query,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(_ARXIV_BASE_URL, params=params)
            response.raise_for_status()
    except Exception:
        return []

    feed = feedparser.parse(response.text)
    entries = feed.get("entries", [])

    papers: list[dict[str, Any]] = []
    for entry in entries:
        links = entry.get("links", [])
        pdf_url = ""
        abstract_url = ""
        for link in links:
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
            if link.get("type") == "text/html":
                abstract_url = link.get("href", "")

        paper_id = entry.get("id", "").split("/abs/")[-1]
        categories = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]

        papers.append(
            {
                "paper_id": paper_id,
                "title": " ".join((entry.get("title") or "").split()),
                "abstract": " ".join((entry.get("summary") or "").split()),
                "year": int(entry.get("published", "0000")[:4]) if entry.get("published") else None,
                "venue": "arXiv",
                "citation_count": 0,
                "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "source": "arxiv",
                "paper_url": abstract_url,
                "pdf_url": pdf_url,
                "categories": categories,
            }
        )
    return papers


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> Any:
    if name == "search_papers":
        query = str(arguments.get("query", "")).strip()
        limit = max(1, min(int(arguments.get("limit", 8)), 20))
        source = str(arguments.get("source", "all")).lower().strip()
        raw_locked = arguments.get("locked_concepts", [])
        locked_concepts = [str(x).strip() for x in raw_locked] if isinstance(raw_locked, list) else []
        locked_concepts = [x for x in locked_concepts if x]

        if source == "semantic_scholar":
            return await _search_semantic_scholar(query, limit, locked_concepts=locked_concepts)
        if source == "arxiv":
            return await _search_arxiv(query, limit, locked_concepts=locked_concepts)

        sem, arx = await asyncio.gather(
            _search_semantic_scholar(query, limit, locked_concepts=locked_concepts),
            _search_arxiv(query, limit, locked_concepts=locked_concepts),
            return_exceptions=True,
        )
        sem_rows = sem if isinstance(sem, list) else []
        arx_rows = arx if isinstance(arx, list) else []
        return sem_rows + arx_rows

    if name == "list_arxiv_categories":
        return {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning",
            "cs.CL": "Computation and Language",
            "stat.ML": "Machine Learning (Statistics)",
            "cs.CV": "Computer Vision",
            "cs.IR": "Information Retrieval",
        }

    raise ValueError(f"Unknown tool: {name}")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        if self.path == "/tools/list":
            self._send_json(
                200,
                {
                    "tools": [
                        {"name": "search_papers", "description": "Search papers by query"},
                        {"name": "list_arxiv_categories", "description": "List common arXiv categories"},
                    ]
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/tools/call":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        body = json.loads(raw.decode("utf-8"))

        name = body.get("name")
        arguments = body.get("arguments", {})
        try:
            result = asyncio.run(handle_tool_call(name, arguments))
            self._send_json(200, {"result": result})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})



def main() -> None:
    parser = argparse.ArgumentParser(description="Local tool server for academic paper search")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Tool server started at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
