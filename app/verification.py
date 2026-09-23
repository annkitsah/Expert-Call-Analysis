"""Deterministic verification of LLM-proposed evidence.

The model proposes ``(segment_id, quote)``. We accept it only if:

1. the segment exists,
2. the segment was spoken by the expert (interviewer text is never evidence),
3. (optionally) the segment belongs to the transcript the claim is about,
4. the quote is a contiguous, verbatim substring of that segment.

Only whitespace, letter case and typographic variants of quotes/dashes are
tolerated. The text we serve and highlight is sliced from the transcript, and
the timestamp is read from the parsed segment - never taken from the model.
"""

from __future__ import annotations

import logging
import re

from app.models import Corpus, Evidence, RawEvidence, VerificationStats

log = logging.getLogger(__name__)

MIN_QUOTE_WORDS = 3
MAX_EVIDENCE_PER_ITEM = 4

_EQUIVALENTS = {
    "'": "['\u2019\u2018]",
    "\u2019": "['\u2019\u2018]",
    "\u2018": "['\u2019\u2018]",
    '"': '["\u201c\u201d]',
    "\u201c": '["\u201c\u201d]',
    "\u201d": '["\u201c\u201d]',
    "-": "[-\u2010\u2011\u2013\u2014]",
    "\u2013": "[-\u2010\u2011\u2013\u2014]",
    "\u2014": "[-\u2010\u2011\u2013\u2014]",
}
_WRAPPING_QUOTES = " \t\r\n\"'\u201c\u201d\u2018\u2019"


def locate_quote(segment_text: str, quote: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` of ``quote`` inside ``segment_text``, or ``None``.

    Matching is case-insensitive and whitespace/typography tolerant, but the words
    themselves must match exactly and be whole words at both ends.
    """
    cleaned = quote.strip(_WRAPPING_QUOTES)
    tokens = cleaned.split()
    if len(tokens) < MIN_QUOTE_WORDS:
        return None
    if "..." in cleaned or "\u2026" in cleaned:
        return None  # ellipsis = stitched fragments; we only accept contiguous text
    pattern = r"\s+".join("".join(_EQUIVALENTS.get(ch, re.escape(ch)) for ch in tok) for tok in tokens)
    match = re.search(rf"(?<!\w){pattern}(?!\w)", segment_text, re.IGNORECASE)
    return (match.start(), match.end()) if match else None


class Verifier:
    """Resolves raw evidence against a corpus and keeps running statistics."""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self.stats = VerificationStats()
        self.warnings: list[str] = []

    def _reject(self, raw: RawEvidence, reason: str) -> None:
        preview = raw.quote.strip().replace("\n", " ")[:80]
        message = f"Dropped a quote cited at {raw.segment_id}: {reason} ({preview!r})"
        log.warning(message)
        self.warnings.append(message)

    def resolve(self, raw: RawEvidence, *, transcript_id: str | None = None) -> Evidence | None:
        self.stats.quotes_checked += 1
        segment = self._corpus.segment(raw.segment_id.strip())
        if segment is None:
            self._reject(raw, "segment does not exist")
            return None
        if not segment.is_expert:
            self._reject(raw, "segment is not expert speech")
            return None
        if transcript_id is not None and segment.transcript_id != transcript_id:
            self._reject(raw, f"segment belongs to {segment.transcript_id}, expected {transcript_id}")
            return None
        span = locate_quote(segment.text, raw.quote)
        if span is None:
            self._reject(raw, "text is not a verbatim excerpt of the segment")
            return None

        transcript = self._corpus.transcript(segment.transcript_id)
        start, end = span
        self.stats.quotes_verified += 1
        return Evidence(
            segment_id=segment.id,
            transcript_id=segment.transcript_id,
            expert_name=transcript.expert_name if transcript else segment.speaker,
            market=transcript.market if transcript else "",
            speaker=segment.speaker,
            timestamp=segment.timestamp,
            seconds=segment.seconds,
            quote=segment.text[start:end],
            start=start,
            end=end,
        )

    def resolve_many(self, raws: list[RawEvidence], *, transcript_id: str | None = None) -> list[Evidence]:
        out: list[Evidence] = []
        seen: set[tuple[str, int, int]] = set()
        for raw in raws:
            ev = self.resolve(raw, transcript_id=transcript_id)
            if ev is None:
                continue
            key = (ev.segment_id, ev.start, ev.end)
            if key in seen:
                continue
            seen.add(key)
            out.append(ev)
            if len(out) >= MAX_EVIDENCE_PER_ITEM:
                break
        return out
