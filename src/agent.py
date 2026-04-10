from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        answer, _ = self.answer_with_sources(question, top_k=top_k)
        return answer

    def answer_with_sources(self, question: str, top_k: int = 3) -> tuple[str, list[dict]]:
        retrieved = self.store.search(question, top_k=top_k)

        context_lines = []
        for index, item in enumerate(retrieved, start=1):
            context_lines.append(f"[{index}] {item['content']}")

        context = "\n\n".join(context_lines) if context_lines else "No relevant context found."
        prompt = (
            "You are a helpful assistant. Answer the user's question using only the context below. "
            "If the context is insufficient, say so briefly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt), retrieved
