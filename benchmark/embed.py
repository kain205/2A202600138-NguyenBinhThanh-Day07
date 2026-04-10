#!/usr/bin/env python3
"""
Pre-compute embeddings for all chunking strategies and save to disk.

Usage:
    python benchmark/embed.py                            # auto-detect PDF/DOCX in benchmark/
    python benchmark/embed.py --file benchmark/tsla-20251231.pdf
    python benchmark/embed.py --embedding openai         # mock | local | openai
    python benchmark/embed.py --chunk-size 400 --overlap 40 --sentences 3

Output: benchmark/embeddings/{fixed,sentence,recursive}.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
)

BENCHMARK_DIR   = Path(__file__).parent
EMBEDDINGS_DIR  = BENCHMARK_DIR / "embeddings"
STRATEGY_LABELS = {"fixed": "Fixed Size", "sentence": "Sentence", "recursive": "Recursive"}

# ── file helpers ──────────────────────────────────────────────────────────────

def auto_detect_file() -> Path | None:
    for pat in ("*.pdf", "*.docx"):
        found = sorted(BENCHMARK_DIR.glob(pat))
        if found:
            return found[0]
    return None


def parse_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = [(pg.extract_text() or "").strip() for pg in reader.pages]
            return "\n\n".join(p for p in pages if p)
        except ImportError:
            pass
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                pages = [(pg.extract_text() or "").strip() for pg in pdf.pages]
            return "\n\n".join(p for p in pages if p)
        except ImportError:
            pass
        print("[ERROR] Install pypdf:  pip install pypdf")
        sys.exit(1)
    if ext == ".docx":
        try:
            from docx import Document as DocxDoc
            return "\n".join(p.text for p in DocxDoc(str(path)).paragraphs if p.text.strip())
        except ImportError:
            print("[ERROR] Install python-docx:  pip install python-docx")
            sys.exit(1)
    return path.read_text(encoding="utf-8", errors="replace").strip()

# ── embedder factory ──────────────────────────────────────────────────────────

def make_embedder(provider: str):
    p = (provider or os.environ.get("EMBEDDING_PROVIDER", "openai")).strip().lower()
    if p == "local":
        try:
            return LocalEmbedder()
        except Exception as e:
            print(f"[WARN] local embedder failed ({e}), falling back to mock")
    elif p == "openai":
        try:
            return OpenAIEmbedder()
        except Exception as e:
            print(f"[WARN] openai embedder failed ({e}), falling back to mock")
    return MockEmbedder()

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings for all chunking strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file",       type=Path, default=None,
                        help="PDF/DOCX/TXT file to index (default: auto-detect in benchmark/)")
    parser.add_argument("--embedding",  choices=["mock", "local", "openai"], default="openai",
                        help="Embedding provider (default: openai)")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap",    type=int, default=40)
    parser.add_argument("--sentences",  type=int, default=3)
    args = parser.parse_args()

    # Resolve file
    file_path = args.file or auto_detect_file()
    if file_path is None:
        print("[ERROR] No file found. Use --file or add a PDF/DOCX to benchmark/")
        sys.exit(1)
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print(f"File     : {file_path}")
    print(f"Embedding: {args.embedding or os.environ.get('EMBEDDING_PROVIDER', 'mock')}")

    # Parse
    print("\nParsing file…")
    text = parse_file(file_path)
    if not text.strip():
        print("[ERROR] File parsed but empty content.")
        sys.exit(1)
    print(f"  {len(text):,} characters")

    # Build embedder
    embedder = make_embedder(args.embedding)
    print(f"Embedder : {embedder._backend_name}")

    # Build chunkers
    chunkers = {
        "fixed":     FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap),
        "sentence":  SentenceChunker(max_sentences_per_chunk=args.sentences),
        "recursive": RecursiveChunker(chunk_size=args.chunk_size),
    }

    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    fname = file_path.name

    for strategy, chunker in chunkers.items():
        label = STRATEGY_LABELS[strategy]
        raw_chunks = chunker.chunk(text)
        print(f"\n[{label}] {len(raw_chunks)} chunks — embedding…")

        docs = [
            Document(
                id=f"{fname}::{strategy}::{i}",
                content=c,
                metadata={"source": fname, "strategy": strategy},
            )
            for i, c in enumerate(raw_chunks)
        ]

        # Embed via EmbeddingStore (reuse existing logic)
        store = EmbeddingStore(collection_name=f"embed_{strategy}", embedding_fn=embedder)

        # Embed in batches with progress
        batch = 50
        for start in range(0, len(docs), batch):
            store.add_documents(docs[start : start + batch])
            done = min(start + batch, len(docs))
            pct  = done / len(docs) * 100
            print(f"  {done}/{len(docs)} ({pct:.0f}%)", end="\r")
        print(f"  {len(docs)}/{len(docs)} (100%) done")

        # Save records to JSON
        out_path = EMBEDDINGS_DIR / f"{strategy}.json"
        payload = {
            "source":     fname,
            "strategy":   strategy,
            "embedder":   embedder._backend_name,
            "chunk_size": args.chunk_size,
            "overlap":    args.overlap,
            "sentences":  args.sentences,
            "chunks": [
                {
                    "id":       rec["id"],
                    "content":  rec["content"],
                    "metadata": rec["metadata"],
                    "embedding": rec["embedding"],
                }
                for rec in store._store
            ],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved → {out_path}")

    print(f"\n✅ All strategies saved to {EMBEDDINGS_DIR}/")
    print("   Run: python benchmark/app.py")


if __name__ == "__main__":
    main()
