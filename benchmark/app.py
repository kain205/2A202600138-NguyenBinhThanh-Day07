#!/usr/bin/env python3
"""
Gradio UI — RAG Benchmark Demo

Prerequisites:
    python benchmark/embed.py          # build embeddings first

Usage:
    python benchmark/app.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import gradio as gr

from src import (
    Document,
    EmbeddingStore,
    KnowledgeBaseAgent,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
)

BENCHMARK_DIR  = Path(__file__).parent
EMBEDDINGS_DIR = BENCHMARK_DIR / "embeddings"
QUERIES_FILE   = BENCHMARK_DIR / "queries.json"

STRATEGIES      = ["fixed", "sentence", "recursive"]
STRATEGY_LABELS = {"fixed": "Fixed Size", "sentence": "Sentence", "recursive": "Recursive"}

# ── load pre-computed embeddings ──────────────────────────────────────────────

def _make_embedder(provider: str):
    p = (provider or "openai").strip().lower()
    if p == "local":
        try:
            return LocalEmbedder()
        except Exception:
            pass
    elif p == "openai":
        try:
            return OpenAIEmbedder()
        except Exception:
            pass
    return MockEmbedder()


def _make_llm():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        try:
            from openai import OpenAI
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
            print(f"[WARN] OpenAI LLM failed: {e}")
    return lambda _: "(no LLM — set OPENAI_API_KEY)"


def load_stores() -> tuple[dict, str]:
    """
    Load pre-computed embeddings from EMBEDDINGS_DIR.
    Returns (stores_dict, status_message).
    """
    if not EMBEDDINGS_DIR.exists():
        return {}, "❌ `benchmark/embeddings/` not found — run `python benchmark/embed.py` first."

    stores: dict = {}
    meta_parts: list[str] = []
    llm = _make_llm()

    for s in STRATEGIES:
        path = EMBEDDINGS_DIR / f"{s}.json"
        if not path.exists():
            meta_parts.append(f"**{STRATEGY_LABELS[s]}**: _(missing)_")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        embedder_name = data.get("embedder", "mock")

        # Infer provider from embedder name
        if "openai" in embedder_name or "text-embedding" in embedder_name:
            provider = "openai"
        elif embedder_name == "mock embeddings fallback":
            provider = "mock"
        else:
            provider = "local"

        embedder = _make_embedder(provider)

        # Reconstruct store by directly loading pre-computed embeddings
        store = EmbeddingStore(collection_name=f"app_{s}", embedding_fn=embedder)
        store._store = [
            {
                "id":        chunk["id"],
                "content":   chunk["content"],
                "metadata":  chunk["metadata"],
                "embedding": chunk["embedding"],
            }
            for chunk in data["chunks"]
        ]

        agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
        stores[s] = {"store": store, "agent": agent, "count": len(data["chunks"])}
        meta_parts.append(f"**{STRATEGY_LABELS[s]}**: {len(data['chunks'])} chunks")

    if not stores:
        status = "❌ No embeddings found — run `python benchmark/embed.py` first."
    else:
        source = next(
            (json.loads((EMBEDDINGS_DIR / f"{s}.json").read_text())["source"]
             for s in STRATEGIES if (EMBEDDINGS_DIR / f"{s}.json").exists()),
            "unknown"
        )
        embedder_name = next(
            (json.loads((EMBEDDINGS_DIR / f"{s}.json").read_text())["embedder"]
             for s in STRATEGIES if (EMBEDDINGS_DIR / f"{s}.json").exists()),
            "unknown"
        )
        status = (
            f"✅ Loaded **{source}** · Embedder: `{embedder_name}`\n\n"
            + "  ·  ".join(meta_parts)
        )

    return stores, status

# ── queries ───────────────────────────────────────────────────────────────────

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

# ── query handler ─────────────────────────────────────────────────────────────

def _build_chunk_md(r: dict, idx: int) -> str:
    score   = r["score"]
    source  = r["metadata"].get("source", "")
    content = textwrap.shorten(r["content"].replace("\n", " "), 500)
    return f"**[{idx}] Score: {score:.4f}** · `{source}`\n\n{content}"


def do_query(
    query_choice: str,
    custom_query: str,
    strategy: str,
    stores: dict | None,
    queries: list[dict],
):
    query_text = (custom_query or "").strip()
    gold = ""

    if query_choice:
        try:
            idx = int(query_choice.split(":")[0][1:]) - 1
        except (ValueError, IndexError):
            idx = -1
        if 0 <= idx < len(queries):
            q_obj = queries[idx]
            if not query_text:
                query_text = q_obj["query"]
            gold = q_obj.get("gold_answer", "")

    def run_one(s: str):
        if not stores or s not in stores:
            return ("", "", "", "")
        if not query_text or query_text.startswith("TODO"):
            return ("", "", "", "")
        info      = stores[s]
        retrieved = info["store"].search(query_text, top_k=3)
        answer    = info["agent"].answer(query_text, top_k=3)
        chunks    = [_build_chunk_md(r, i + 1) for i, r in enumerate(retrieved[:3])]
        while len(chunks) < 3:
            chunks.append("")
        return (answer, chunks[0], chunks[1], chunks[2])

    if not stores:
        warn   = "⚠️ Embeddings not loaded — run `python benchmark/embed.py` first."
        r_empty = (warn, "", "", "")
    else:
        r_empty = ("", "", "", "")

    is_all3 = strategy == "All 3"
    vis_f   = is_all3 or strategy == "fixed"
    vis_s   = is_all3 or strategy == "sentence"
    vis_r   = is_all3 or strategy == "recursive"

    r_fixed     = run_one("fixed")     if vis_f else r_empty
    r_sentence  = run_one("sentence")  if vis_s else r_empty
    r_recursive = run_one("recursive") if vis_r else r_empty

    if not stores:
        if vis_f: r_fixed     = r_empty
        if vis_s: r_sentence  = r_empty
        if vis_r: r_recursive = r_empty

    return (
        gold,
        gr.update(visible=vis_f),    *r_fixed,
        gr.update(visible=vis_s),    *r_sentence,
        gr.update(visible=vis_r),    *r_recursive,
    )


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

# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    queries  = load_queries()
    q_labels = query_dropdown_choices(queries)
    stores, load_status = load_stores()

    with gr.Blocks(title="RAG Benchmark Demo", theme=gr.themes.Soft()) as demo:
        state_stores  = gr.State(stores)
        state_queries = gr.State(queries)

        gr.Markdown("# RAG Benchmark Demo")
        gr.Markdown(load_status)

        gr.Markdown("---")

        with gr.Group():
            gr.Markdown("### Query")
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
                label="Query text (auto-filled — edit freely)",
                placeholder="Type a custom query…",
                lines=2,
            )
            run_btn = gr.Button("▶  Run Query", variant="primary")

        gr.Markdown("---")
        gr.Markdown("### Results")

        gold_box = gr.Textbox(label="Gold Answer", interactive=False, lines=3)

        with gr.Row():
            with gr.Column(visible=True) as fixed_col:
                gr.Markdown("#### Fixed Size")
                fixed_answer = gr.Textbox(label="AI Answer", interactive=False, lines=4)
                with gr.Accordion("Top 3 Chunks", open=True):
                    fixed_c1 = gr.Markdown()
                    gr.Markdown("---")
                    fixed_c2 = gr.Markdown()
                    gr.Markdown("---")
                    fixed_c3 = gr.Markdown()

            with gr.Column(visible=False) as sentence_col:
                gr.Markdown("#### Sentence")
                sentence_answer = gr.Textbox(label="AI Answer", interactive=False, lines=4)
                with gr.Accordion("Top 3 Chunks", open=True):
                    sentence_c1 = gr.Markdown()
                    gr.Markdown("---")
                    sentence_c2 = gr.Markdown()
                    gr.Markdown("---")
                    sentence_c3 = gr.Markdown()

            with gr.Column(visible=False) as recursive_col:
                gr.Markdown("#### Recursive")
                recursive_answer = gr.Textbox(label="AI Answer", interactive=False, lines=4)
                with gr.Accordion("Top 3 Chunks", open=True):
                    recursive_c1 = gr.Markdown()
                    gr.Markdown("---")
                    recursive_c2 = gr.Markdown()
                    gr.Markdown("---")
                    recursive_c3 = gr.Markdown()

        # Wiring
        query_dropdown.change(
            fn=on_query_select,
            inputs=[query_dropdown, state_queries],
            outputs=[query_text],
        )

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


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
