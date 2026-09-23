"""Context selection for cross-transcript Q&A.

* Small corpus (the 3-call case): send everything. Retrieval can only lose
  information, and the whole corpus is ~5k tokens.
* Large corpus (30+ calls): score segments with BM25 and send the top-k plus their
  neighbouring turns. BM25 is lexical, so it can miss paraphrases; the production
  upgrade is hybrid lexical + embedding retrieval with a reranker (see README).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from app.models import Corpus, Segment

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by do does for from had has have how i if in into is it its of on or "
    "our so than that the their them then there these they this to was we were what when where which "
    "who why will with would you your about across any between did say said tell".split()
)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token for English)."""
    return max(1, len(text) // 4)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Bm25Index:
    def __init__(self, segments: list[Segment], k1: float = 1.5, b: float = 0.75) -> None:
        self._segments = segments
        self._k1, self._b = k1, b
        self._docs = [Counter(tokenize(s.text)) for s in segments]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(segments)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query: str, k: int) -> list[tuple[Segment, float]]:
        terms = tokenize(query)
        scored: list[tuple[Segment, float]] = []
        for seg, doc, length in zip(self._segments, self._docs, self._lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                norm = tf + self._k1 * (1 - self._b + self._b * length / (self._avg_len or 1.0))
                score += self._idf.get(term, 0.0) * tf * (self._k1 + 1) / norm
            if score > 0:
                scored.append((seg, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


@dataclass(frozen=True)
class ContextSelection:
    mode: Literal["full", "retrieval"]
    segments_by_transcript: dict[str, list[Segment]]

    @property
    def segment_count(self) -> int:
        return sum(len(v) for v in self.segments_by_transcript.values())


def select_context(
    corpus: Corpus,
    query: str,
    *,
    token_budget: int,
    top_k: int,
    neighbours: int,
) -> ContextSelection:
    total = sum(estimate_tokens(s.text) for t in corpus.transcripts for s in t.segments)
    if total <= token_budget:
        return ContextSelection("full", {t.id: list(t.segments) for t in corpus.transcripts})

    all_segments = [s for t in corpus.transcripts for s in t.segments]
    hits = Bm25Index(all_segments).top(query, top_k)
    wanted: dict[str, set[int]] = {t.id: set() for t in corpus.transcripts}
    for seg, _score in hits:
        for idx in range(seg.index - neighbours, seg.index + neighbours + 1):
            wanted[seg.transcript_id].add(idx)

    selected: dict[str, list[Segment]] = {}
    for t in corpus.transcripts:
        keep = [s for s in t.segments if s.index in wanted[t.id]]
        selected[t.id] = keep  # possibly empty: the transcript is still listed, with no excerpts
    return ContextSelection("retrieval", selected)
