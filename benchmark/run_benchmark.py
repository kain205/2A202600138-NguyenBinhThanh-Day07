#!/usr/bin/env python3
"""
Benchmark script — output markdown tables để paste vào REPORT.md.

── Chế độ 1: Baseline Analysis (section 3) ──────────────────────────────────
  python benchmark/run_benchmark.py --baseline

  → In bảng so sánh 3 strategies trên từng tài liệu trong data/
    Paste vào: ## 3. Chunking Strategy > Baseline Analysis

── Chế độ 2: Kết Quả Cá Nhân (section 6) ───────────────────────────────────
  python benchmark/run_benchmark.py --name Alice --strategy sentence
  python benchmark/run_benchmark.py --name Bob   --strategy fixed   --chunk-size 300
  python benchmark/run_benchmark.py --name Carol --strategy recursive

  → In bảng query results + lưu benchmark/results/<name>.json
    Paste vào: ## 6. Results > Kết Quả Của Tôi

── Chế độ 3: So Sánh Nhóm (section 3 cuối) ─────────────────────────────────
  python benchmark/run_benchmark.py --compare

  → In bảng So Sánh Với Thành Viên Khác (đọc benchmark/results/*.json)
    Paste vào: ## 3. Chunking Strategy > So Sánh Với Thành Viên Khác

Strategies: fixed | sentence | recursive
Embedding:  --embedding mock (default) | local | openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
)

QUERIES_FILE = Path(__file__).parent / "queries.json"
RESULTS_DIR  = Path(__file__).parent / "results"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_docs(data_dir: Path) -> list[Document]:
    docs = []
    for p in sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.md")):
        if p.name.startswith(".") or p.name == "REPORT.md":
            continue
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            docs.append(Document(id=p.stem, content=text,
                                 metadata={"source": p.name}))
    return docs


def load_queries() -> list[dict]:
    if not QUERIES_FILE.exists():
        print(f"[ERROR] {QUERIES_FILE} not found — điền queries trước.")
        sys.exit(1)
    qs = json.loads(QUERIES_FILE.read_text(encoding="utf-8")).get("queries", [])
    if any(q["query"].startswith("TODO") for q in qs):
        print("[WARN] queries.json còn TODO — nhóm chưa điền đủ queries.")
    return qs


def make_chunker(strategy: str, chunk_size: int, overlap: int, sentences: int):
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=sentences)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    raise ValueError(f"Unknown strategy '{strategy}'")


def make_embedder(provider: str | None):
    p = (provider or os.environ.get("EMBEDDING_PROVIDER", "mock")).strip().lower()
    if p == "local":
        try: return LocalEmbedder()
        except Exception as e: print(f"[WARN] local embedder failed ({e}), dùng mock")
    elif p == "openai":
        try: return OpenAIEmbedder()
        except Exception as e: print(f"[WARN] openai embedder failed ({e}), dùng mock")
    return MockEmbedder()


def make_llm():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        try:
            from openai import OpenAI
            client = OpenAI()
            def _fn(prompt: str) -> str:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                )
                return r.choices[0].message.content.strip()
            return _fn
        except Exception as e:
            print(f"[WARN] OpenAI LLM failed ({e})")
    return lambda _: "(no LLM — set OPENAI_API_KEY)"


def short(text: str, n: int = 55) -> str:
    return textwrap.shorten(text.replace("\n", " "), n, placeholder="…")

# ── mode 1: baseline ──────────────────────────────────────────────────────────

def cmd_baseline(args: argparse.Namespace) -> None:
    docs = load_docs(Path(args.data_dir))
    if not docs:
        print(f"[ERROR] Không có file .txt/.md trong {args.data_dir}/")
        sys.exit(1)

    comparator = ChunkingStrategyComparator()

    rows: list[tuple] = []   # (doc_name, strategy_label, count, avg_len)
    strategy_keys = [
        ("FixedSizeChunker (`fixed_size`)", "fixed_size"),
        ("SentenceChunker (`by_sentences`)", "by_sentences"),
        ("RecursiveChunker (`recursive`)", "recursive"),
    ]

    for doc in docs[:3]:   # tối đa 3 docs như hướng dẫn
        result = comparator.compare(doc.content, chunk_size=args.chunk_size)
        for label, key in strategy_keys:
            r = result[key]
            rows.append((doc.id, label, r["count"], round(r["avg_length"])))

    # Print markdown table matching REPORT.md section 3 Baseline Analysis
    print("\n<!-- Paste vào: ## 3. Chunking Strategy > Baseline Analysis -->\n")
    print("| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |")
    print("|-----------|----------|-------------|------------|-------------------|")
    prev_doc = None
    for doc_name, label, count, avg in rows:
        doc_cell = doc_name if doc_name != prev_doc else ""
        prev_doc = doc_name
        print(f"| {doc_cell} | {label} | {count} | {avg} | (tự đánh giá) |")
    print()

# ── mode 2: benchmark ─────────────────────────────────────────────────────────

def cmd_benchmark(args: argparse.Namespace) -> None:
    queries = load_queries()
    docs    = load_docs(Path(args.data_dir))
    if not docs:
        print(f"[ERROR] Không có file .txt/.md trong {args.data_dir}/")
        sys.exit(1)

    chunker = make_chunker(args.strategy, args.chunk_size, args.overlap, args.sentences)
    chunks: list[Document] = []
    for doc in docs:
        for i, c in enumerate(chunker.chunk(doc.content)):
            chunks.append(Document(
                id=f"{doc.id}::c{i}", content=c,
                metadata={"source": doc.metadata["source"], "parent": doc.id},
            ))

    embedder = make_embedder(args.embedding)
    store = EmbeddingStore(embedding_fn=embedder)
    store.add_documents(chunks)
    agent = KnowledgeBaseAgent(store=store, llm_fn=make_llm())

    print(f"\n[{args.name}] strategy={args.strategy}, chunks={len(chunks)}, embed={embedder._backend_name}\n")

    # Print queries table (gold answers) for section 6 top
    print("<!-- Paste vào: ## 6. Results > Benchmark Queries & Gold Answers -->\n")
    print("| # | Query | Gold Answer |")
    print("|---|-------|-------------|")
    for i, q in enumerate(queries, 1):
        print(f"| {i} | {short(q['query'], 60)} | {short(q.get('gold_answer',''), 60)} |")

    # Run and collect results
    records = []
    print()
    for i, q in enumerate(queries, 1):
        retrieved = store.search(q["query"], top_k=3)
        answer    = agent.answer(q["query"], top_k=3)
        top1 = retrieved[0] if retrieved else None
        records.append({
            "query": q["query"],
            "gold_answer": q.get("gold_answer", ""),
            "top1_source": top1["metadata"].get("source", "") if top1 else "",
            "top1_score": round(top1["score"], 4) if top1 else 0,
            "top1_preview": top1["content"][:200] if top1 else "",
            "top3": [{"score": r["score"], "source": r["metadata"].get("source",""),
                      "preview": r["content"][:200]} for r in retrieved],
            "agent_answer": answer,
            "relevant": "",  # student fills manually
        })

    # Print results table for section 6
    print("\n<!-- Paste vào: ## 6. Results > Kết Quả Của Tôi -->\n")
    print("| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |")
    print("|---|-------|--------------------------------|-------|-----------|------------------------|")
    for i, r in enumerate(records, 1):
        print(f"| {i} | {short(r['query'],40)} | {short(r['top1_preview'],45)} "
              f"| {r['top1_score']} |  | {short(r['agent_answer'],50)} |")
    print()

    # Save JSON for --compare
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps({
        "name": args.name,
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "sentences": args.sentences,
        "embedding": embedder._backend_name,
        "chunk_count": len(chunks),
        "queries": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {out}  (dùng cho --compare)\n")

# ── mode 3: compare ───────────────────────────────────────────────────────────

def cmd_compare() -> None:
    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    if not files:
        print(f"[ERROR] Không có file JSON trong {RESULTS_DIR}/")
        sys.exit(1)

    members = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    # Print "So Sánh Với Thành Viên Khác" table for section 3
    print("\n<!-- Paste vào: ## 3. Chunking Strategy > So Sánh Với Thành Viên Khác -->\n")
    print("| Thành viên | Strategy | Chunks | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |")
    print("|-----------|----------|--------|----------------------|-----------|----------|")
    for m in members:
        score = sum(q.get("manual_score", 0) for q in m["queries"])
        max_s = len(m["queries"]) * 2
        print(f"| {m['name']} | {m['strategy']} | {m['chunk_count']} | {score}/{max_s} "
              f"| (tự điền) | (tự điền) |")
    print()

    # Per-query comparison
    n = max(len(m["queries"]) for m in members)
    print("<!-- Per-query scores -->\n")
    header = "| Query |" + "".join(f" {m['name']} |" for m in members)
    print(header)
    print("|" + "---|" * (len(members) + 1))
    for qi in range(n):
        q_text = short(members[0]["queries"][qi]["query"], 35) if qi < len(members[0]["queries"]) else f"Q{qi+1}"
        row = f"| {q_text} |"
        for m in members:
            sc = m["queries"][qi].get("manual_score", "-") if qi < len(m["queries"]) else "-"
            row += f" {sc} |"
        print(row)
    print()

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark — output markdown cho REPORT.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python benchmark/run_benchmark.py --baseline
          python benchmark/run_benchmark.py --name Alice --strategy sentence
          python benchmark/run_benchmark.py --name Bob   --strategy fixed --chunk-size 300
          python benchmark/run_benchmark.py --compare
        """),
    )
    p.add_argument("--baseline",    action="store_true", help="In bảng Baseline Analysis (section 3)")
    p.add_argument("--compare",     action="store_true", help="In bảng So Sánh Nhóm (section 3 cuối)")
    p.add_argument("--name",        help="Tên bạn (dùng cho tên file JSON kết quả)")
    p.add_argument("--strategy",    choices=["fixed","sentence","recursive"], default="fixed")
    p.add_argument("--chunk-size",  type=int, default=400)
    p.add_argument("--overlap",     type=int, default=40)
    p.add_argument("--sentences",   type=int, default=3)
    p.add_argument("--embedding",   choices=["mock","local","openai"], default=None)
    p.add_argument("--data-dir",    default="data")
    args = p.parse_args()

    if args.baseline:
        cmd_baseline(args)
    elif args.compare:
        cmd_compare()
    elif args.name:
        cmd_benchmark(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
