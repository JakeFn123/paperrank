from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List

import fitz
import httpx

from app.config import CHROMA_DIR, PDF_DIR
from app.llm import LLMGateway
from app.schemas import EvidenceSpan, PaperRecord


class LocalVectorStore:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_file = self.root / "chunks.json"
        self.rows: List[dict] = []
        self._load()

    def _load(self) -> None:
        if self.db_file.exists():
            try:
                self.rows = json.loads(self.db_file.read_text(encoding="utf-8"))
            except Exception:
                self.rows = []

    def _save(self) -> None:
        self.db_file.write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")

    def upsert(self, chunks: List[dict]) -> None:
        existing_ids = {r["id"] for r in self.rows}
        for c in chunks:
            if c["id"] in existing_ids:
                continue
            self.rows.append(c)
        self._save()

    def query(self, query_embedding: List[float], paper_id: str, top_k: int) -> List[dict]:
        candidates = [r for r in self.rows if r.get("paper_id") == paper_id]
        scored = []
        for row in candidates:
            sim = cosine_similarity(query_embedding, row.get("embedding", []))
            scored.append((sim, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class RAGEngine:
    def __init__(self, llm: LLMGateway) -> None:
        self.llm = llm
        self.store = LocalVectorStore(CHROMA_DIR)

    def ingest_paper_pdf(self, paper: PaperRecord) -> int:
        if not paper.pdf_url:
            return 0
        pdf_path = self._download_pdf(paper.paper_id, paper.pdf_url)
        if not pdf_path:
            return 0

        chunks = self._extract_chunks(pdf_path)
        if not chunks:
            return 0

        embeddings = self.llm.embed_texts([c["text"] for c in chunks])
        rows = []
        for idx, c in enumerate(chunks):
            rows.append(
                {
                    "id": c["id"],
                    "paper_id": paper.paper_id,
                    "page": c["page"],
                    "text": c["text"],
                    "embedding": embeddings[idx],
                }
            )
        self.store.upsert(rows)
        return len(rows)

    def retrieve_evidence(self, question: str, paper: PaperRecord, top_k: int = 3) -> List[EvidenceSpan]:
        query_emb = self.llm.embed_texts([question])[0]
        rows = self.store.query(query_emb, paper.paper_id, top_k)

        evidence: List[EvidenceSpan] = []
        for row in rows:
            evidence.append(
                EvidenceSpan(
                    paper_id=paper.paper_id,
                    ref_id="",
                    page=int(row.get("page", 1)),
                    text=str(row.get("text", "")),
                )
            )
        return evidence

    def _download_pdf(self, paper_id: str, pdf_url: str) -> Path | None:
        out = Path(PDF_DIR) / f"{paper_id}.pdf"
        if out.exists() and out.stat().st_size > 0:
            return out

        try:
            resp = httpx.get(pdf_url, timeout=45.0, follow_redirects=True)
            resp.raise_for_status()
            out.write_bytes(resp.content)
            return out
        except Exception:
            return None

    def _extract_chunks(self, pdf_path: Path, chunk_size: int = 1300, overlap: int = 180) -> List[dict]:
        chunks: List[dict] = []
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return chunks

        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            start = 0
            while start < len(text):
                part = text[start : start + chunk_size]
                if part:
                    chunks.append(
                        {
                            "id": f"{pdf_path.stem}-p{page_idx}-{start}",
                            "page": page_idx,
                            "text": part,
                        }
                    )
                start += max(1, chunk_size - overlap)
        return chunks
