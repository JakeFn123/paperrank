from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_BACKEND = os.getenv("LLM_BACKEND", "mock").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

CHROMA_DIR = str((BASE_DIR / os.getenv("CHROMA_DIR", "data/chroma")).resolve())
PDF_DIR = str((BASE_DIR / os.getenv("PDF_DIR", "data/pdfs")).resolve())

SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))
SUBQUERY_COUNT = int(os.getenv("SUBQUERY_COUNT", "3"))

MCP_SERVER_PATH = str((BASE_DIR / "mcp_servers" / "academic_search_server.py").resolve())

for path in [CHROMA_DIR, PDF_DIR]:
    Path(path).mkdir(parents=True, exist_ok=True)
