from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", text.strip()) if s.strip()]
        if not sentences:
            return [text.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            part = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if part:
                chunks.append(part)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        chunks = self._split(text.strip(), self.separators)
        return [c for c in chunks if c]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        text = current_text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        if not remaining_separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = remaining_separators[0]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split(text, remaining_separators[1:])

        packed_parts: list[str] = []
        current = ""

        for raw_part in parts:
            part = raw_part.strip()
            if not part:
                continue

            candidate = part if not current else f"{current}{separator}{part}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                packed_parts.append(current)

            if len(part) <= self.chunk_size:
                current = part
            else:
                packed_parts.extend(self._split(part, remaining_separators[1:]))
                current = ""

        if current:
            packed_parts.append(current)

        results: list[str] = []
        for part in packed_parts:
            if len(part) <= self.chunk_size:
                results.append(part)
            else:
                results.extend(self._split(part, remaining_separators[1:]))
        return results


class ParagraphChunker:
    """
    Split text by grouping complete paragraphs up to max_chunk_size characters.

    A paragraph is a block of text separated by one or more blank lines.
    Paragraphs are greedily packed until the next one would exceed the limit.
    Paragraphs longer than max_chunk_size are split by sentences as fallback.
    """

    _PARA_SPLIT = re.compile(r"\n\s*\n")
    _SENT_SPLIT = re.compile(r"(?<=[.!?])(?:\s+|\n+)")

    def __init__(self, max_chunk_size: int = 800) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in self._PARA_SPLIT.split(text.strip()) if p.strip()]
        if not paragraphs:
            return [text.strip()]

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            if len(para) > self.max_chunk_size:
                sentences = [s.strip() for s in self._SENT_SPLIT.split(para) if s.strip()]
                # Carry over any accumulated short paragraphs into the first sentence chunk
                # so headers like "Tesla, Inc." don't become standalone noise chunks.
                sub: list[str] = []
                sub_len = 0
                if current_parts:
                    seed = "\n\n".join(current_parts)
                    sub = [seed]
                    sub_len = len(seed)
                    current_parts = []
                    current_len = 0
                for sent in sentences:
                    if sub and sub_len + len(sent) + 1 > self.max_chunk_size:
                        chunks.append(" ".join(sub))
                        sub = [sent]
                        sub_len = len(sent)
                    else:
                        sub.append(sent)
                        sub_len += len(sent) + 1
                if sub:
                    chunks.append(" ".join(sub))
                continue

            added_len = len(para) + (2 if current_parts else 0)
            if current_parts and current_len + added_len > self.max_chunk_size:
                chunks.append("\n\n".join(current_parts))
                current_parts = [para]
                current_len = len(para)
            else:
                current_parts.append(para)
                current_len += added_len

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10)),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict[str, dict] = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = (sum(len(chunk) for chunk in chunks) / count) if count else 0.0
            comparison[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }
        return comparison
