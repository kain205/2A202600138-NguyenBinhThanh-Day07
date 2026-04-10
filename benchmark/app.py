#!/usr/bin/env python3
"""
Gradio UI — RAG Benchmark Demo

Usage:
    python benchmark/app.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr

from src import (
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

# ── constants ─────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent
DATA_DIR      = ROOT / "data"
QUERIES_FILE  = BENCHMARK_DIR / "queries.json"

STRATEGIES      = ["fixed", "sentence", "recursive"]
STRATEGY_LABELS = {"fixed": "Fixed Size", "sentence": "Sentence", "recursive": "Recursive"}

# ── file helpers ──────────────────────────────────────────────────────────────

def discover_files() -> list[str]:
    """Return list of (display_label, full_path) for all supported files."""
    files: list[str] = []
    for pat in ("*.pdf", "*.docx"):
        files.extend(str(p) for p in sorted(BENCHMARK_DIR.glob(pat)))
    for pat in ("*.txt", "*.md"):
        files.extend(
            str(p) for p in sorted(DATA_DIR.glob(pat))
            if not p.name.startswith(".") and p.name not in ("REPORT.md",)
        )
    return files


def _parse_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n\n".join(
            (pg.extract_text() or "").strip()
            for pg in reader.pages
            if (pg.extract_text() or "").strip()
        )
    except ImportError:
        pass
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(path) as pdf:
            return "\n\n".join(
                (pg.extract_text() or "").strip()
                for pg in pdf.pages
                if (pg.extract_text() or "").strip()
            )
    except ImportError:
        pass
    return "[ERROR] Install pypdf to parse PDF:  pip install pypdf"


def _parse_docx(path: str) -> str:
    try:
        from docx import Document as DocxDoc  # type: ignore
        doc = DocxDoc(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        return "[ERROR] Install python-docx:  pip install python-docx"


def parse_file(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    return p.read_text(encoding="utf-8", errors="replace").strip()

# ── embedder / LLM factories ──────────────────────────────────────────────────

def make_embedder(provider: str):
    p = (provider or "mock").strip().lower()
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


def make_llm():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            def _call(prompt: str) -> str:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                return r.choices[0].message.content.strip()
            return _call
        except Exception as e:
            print(f"[WARN] OpenAI LLM failed ({e})")
    return lambda _: "(no LLM — set OPENAI_API_KEY to enable real answers)"

# ── query helpers ─────────────────────────────────────────────────────────────

def load_queries() -> list[dict]:
    if not QUERIES_FILE.exists():
        return []
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8")).get("queries", [])


def query_dropdown_choices(queries: list[dict]) -> list[str]:
    choices = []
    for i, q in enumerate(queries, 1):
        text = q["query"]
        if text.startswith("TODO"):
            choices.append(f"Q{i}: [TODO — fill in queries.json]")
        else:
            choices.append(f"Q{i}: {textwrap.shorten(text, 88)}")
    return choices

# ── sync / index ──────────────────────────────────────────────────────────────

def do_sync(
    dropdown_file: str,
    upload_file,           # gr.File value — has .name attribute or None
    embed_provider: str,
    chunk_size: int,
    overlap: int,
    sentences: int,
):
    # Resolve which file to use
    file_path: str | None = None
    if upload_file is not None:
        # Gradio File returns a temp path string (or NamedString)
        file_path = upload_file if isinstance(upload_file, str) else upload_file.name
    elif dropdown_file:
        file_path = dropdown_file

    if not file_path:
        return None, "⚠️ Select a file from the dropdown or upload one."

    text = parse_file(file_path)
    if text.startswith("[ERROR]"):
        return None, text
    if not text.strip():
        return None, "⚠️ File parsed but returned empty content."

    embedder = make_embedder(embed_provider)
    llm      = make_llm()
    fname    = Path(file_path).name

    chunkers: dict[str, object] = {
        "fixed":     FixedSizeChunker(chunk_size=int(chunk_size), overlap=int(overlap)),
        "sentence":  SentenceChunker(max_sentences_per_chunk=int(sentences)),
        "recursive": RecursiveChunker(chunk_size=int(chunk_size)),
    }

    stores: dict = {}
    summary_parts: list[str] = []
    for s, chunker in chunkers.items():
        raw = chunker.chunk(text)
        docs = [
            Document(
                id=f"{fname}::{s}::{i}",
                content=c,
                metadata={"source": fname, "strategy": s},
            )
            for i, c in enumerate(raw)
        ]
        store = EmbeddingStore(collection_name=f"demo_{s}", embedding_fn=embedder)
        store.add_documents(docs)
        agent  = KnowledgeBaseAgent(store=store, llm_fn=llm)
        stores[s] = {"store": store, "agent": agent, "count": len(docs)}
        summary_parts.append(f"**{STRATEGY_LABELS[s]}**: {len(docs)} chunks")

    status = (
        f"✅ Indexed **{fname}**  |  Embedder: `{embedder._backend_name}`\n\n"
        + "  ·  ".join(summary_parts)
    )
    return stores, status

# ── run query ─────────────────────────────────────────────────────────────────

def _build_chunk_text(r: dict, idx: int) -> str:
    score   = r["score"]
    source  = r["metadata"].get("source", "")
    content = r["content"].replace("\n", " ")
    short   = textwrap.shorten(content, 500)
    return f"**[{idx}] Score: {score:.4f}** | `{source}`\n\n{short}"


def do_query(
    query_choice: str,
    custom_query: str,
    strategy: str,
    stores,           # dict or None (from gr.State)
    queries: list[dict],
):
    # Resolve query text + gold answer
    query_text = (custom_query or "").strip()
    gold       = ""

    if query_choice:
        try:
            q_idx = int(query_choice.split(":")[0][1:]) - 1
        except (ValueError, IndexError):
            q_idx = -1
        if 0 <= q_idx < len(queries):
            q_obj = queries[q_idx]
            if not query_text:
                query_text = q_obj["query"]
            gold = q_obj.get("gold_answer", "")

    # Helper: build (answer, c1, c2, c3) for one strategy
    def run_one(s: str):
        if not stores or s not in stores:
            return ("", "", "", "")
        if not query_text or query_text.startswith("TODO"):
            return ("", "", "", "")
        info       = stores[s]
        retrieved  = info["store"].search(query_text, top_k=3)
        answer     = info["agent"].answer(query_text, top_k=3)
        chunks     = [_build_chunk_text(r, i + 1) for i, r in enumerate(retrieved[:3])]
        while len(chunks) < 3:
            chunks.append("")
        return (answer, chunks[0], chunks[1], chunks[2])

    if not stores:
        warn = "⚠️ Index a document first — click **Sync Embeddings** above."
        r_empty = (warn, "", "", "")
    else:
        r_empty = ("", "", "", "")

    is_all3 = strategy == "All 3"
    vis_f   = is_all3 or strategy == "fixed"
    vis_s   = is_all3 or strategy == "sentence"
    vis_r   = is_all3 or strategy == "recursive"

    r_fixed     = run_one("fixed")
    r_sentence  = run_one("sentence")
    r_recursive = run_one("recursive")

    # If stores missing, push warning to the visible column(s)
    if not stores:
        if vis_f:   r_fixed     = r_empty
        if vis_s:   r_sentence  = r_empty
        if vis_r:   r_recursive = r_empty

    return (
        gold,
        gr.update(visible=vis_f),   *r_fixed,
        gr.update(visible=vis_s),   *r_sentence,
        gr.update(visible=vis_r),   *r_recursive,
    )

# ── update query text when dropdown changes ───────────────────────────────────

def on_query_select(choice: str, queries: list[dict]) -> str:
    if not choice:
        return ""
    try:
        idx = int(choice.split(":")[0][1:]) - 1
    except (ValueError, IndexError):
        return ""
    if 0 <= idx < len(queries):
        q = queries[idx]["query"]
        return "" if q.startswith("TODO") else q
    return ""

# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    all_files = discover_files()
    queries   = load_queries()
    q_labels  = query_dropdown_choices(queries)

    # Default file — prefer tsla PDF if present
    default_file = next(
        (f for f in all_files if "tsla" in Path(f).name.lower()),
        (all_files[0] if all_files else None),
    )

    file_labels = {str(p): Path(p).name for p in all_files}

    with gr.Blocks(title="RAG Benchmark Demo", theme=gr.themes.Soft()) as demo:
        # ── persistent state ──────────────────────────────────────────────────
        state_stores  = gr.State(None)     # dict of {strategy: {store, agent, count}}
        state_queries = gr.State(queries)  # list of query dicts

        # ── header ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
# RAG Benchmark Demo
Compare **Fixed Size**, **Sentence**, and **Recursive** chunking strategies on your documents.
"""
        )

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 1 — Sync Embeddings
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Accordion("Step 1 — Sync Embeddings", open=True):
            gr.Markdown("Index a document into all three chunking strategies at once.")

            with gr.Row():
                file_dropdown = gr.Dropdown(
                    choices=[(Path(p).name, p) for p in all_files],
                    value=default_file,
                    label="Select file (auto-scanned from benchmark/ and data/)",
                    scale=3,
                )
                upload_file = gr.File(
                    label="Or upload PDF / DOCX",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    scale=2,
                )

            with gr.Row():
                embed_radio = gr.Radio(
                    choices=["mock", "local", "openai"],
                    value="mock",
                    label="Embedding provider",
                    info="mock = fast deterministic | local = sentence-transformers | openai = API key needed",
                )

            with gr.Row():
                chunk_size_sl  = gr.Slider(100, 1000, value=400, step=50,
                                           label="Chunk size (chars)  [fixed / recursive]")
                overlap_sl     = gr.Slider(0, 200,  value=40,  step=10,
                                           label="Overlap (chars)  [fixed only]")
                sentences_sl   = gr.Slider(1, 10,   value=3,   step=1,
                                           label="Sentences per chunk  [sentence only]")

            sync_btn    = gr.Button("🔄  Sync Embeddings", variant="primary")
            sync_status = gr.Markdown("_Not indexed yet._")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 2 — Run Query
        # ═══════════════════════════════════════════════════════════════════════
        gr.Markdown("---")
        with gr.Group():
            gr.Markdown("### Step 2 — Run Benchmark Query")

            with gr.Row():
                query_dropdown = gr.Dropdown(
                    choices=q_labels,
                    value=q_labels[0] if q_labels else None,
                    label="Preset benchmark queries",
                    scale=4,
                )
                strategy_radio = gr.Radio(
                    choices=["fixed", "sentence", "recursive", "All 3"],
                    value="fixed",
                    label="Chunking strategy",
                    scale=2,
                )

            query_text = gr.Textbox(
                label="Query (auto-filled from dropdown — edit freely)",
                placeholder="Type or paste a custom query…",
                lines=2,
            )

            run_btn = gr.Button("▶  Run Query", variant="primary")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 3 — Results
        # ═══════════════════════════════════════════════════════════════════════
        gr.Markdown("---")
        gr.Markdown("### Results")

        gold_box = gr.Textbox(
            label="Gold Answer (from queries.json)",
            interactive=False,
            lines=3,
        )

        with gr.Row():
            # ── Fixed ─────────────────────────────────────────────────────────
            with gr.Column(visible=True) as fixed_col:
                gr.Markdown("#### Fixed Size Chunking")
                fixed_answer = gr.Textbox(label="AI Answer", interactive=False, lines=4)
                with gr.Accordion("Top 3 Retrieved Chunks", open=True):
                    fixed_c1 = gr.Markdown()
                    gr.Markdown("---")
                    fixed_c2 = gr.Markdown()
                    gr.Markdown("---")
                    fixed_c3 = gr.Markdown()

            # ── Sentence ──────────────────────────────────────────────────────
            with gr.Column(visible=False) as sentence_col:
                gr.Markdown("#### Sentence Chunking")
                sentence_answer = gr.Textbox(label="AI Answer", interactive=False, lines=4)
                with gr.Accordion("Top 3 Retrieved Chunks", open=True):
                    sentence_c1 = gr.Markdown()
                    gr.Markdown("---")
                    sentence_c2 = gr.Markdown()
                    gr.Markdown("---")
                    sentence_c3 = gr.Markdown()

            # ── Recursive ─────────────────────────────────────────────────────
            with gr.Column(visible=False) as recursive_col:
                gr.Markdown("#### Recursive Chunking")
                recursive_answer = gr.Textbox(label="AI Answer", interactive=False, lines=4)
                with gr.Accordion("Top 3 Retrieved Chunks", open=True):
                    recursive_c1 = gr.Markdown()
                    gr.Markdown("---")
                    recursive_c2 = gr.Markdown()
                    gr.Markdown("---")
                    recursive_c3 = gr.Markdown()

        # ═══════════════════════════════════════════════════════════════════════
        # Wiring
        # ═══════════════════════════════════════════════════════════════════════

        # Sync button
        sync_btn.click(
            fn=do_sync,
            inputs=[file_dropdown, upload_file, embed_radio, chunk_size_sl, overlap_sl, sentences_sl],
            outputs=[state_stores, sync_status],
        )

        # Populate query text when user picks from dropdown
        query_dropdown.change(
            fn=on_query_select,
            inputs=[query_dropdown, state_queries],
            outputs=[query_text],
        )

        # Run query
        run_btn.click(
            fn=do_query,
            inputs=[query_dropdown, query_text, strategy_radio, state_stores, state_queries],
            outputs=[
                gold_box,
                fixed_col,    fixed_answer,    fixed_c1,    fixed_c2,    fixed_c3,
                sentence_col, sentence_answer, sentence_c1, sentence_c2, sentence_c3,
                recursive_col,recursive_answer,recursive_c1,recursive_c2,recursive_c3,
            ],
        )

    return demo


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
