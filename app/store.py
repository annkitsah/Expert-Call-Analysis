"""In-memory corpus + on-disk analysis cache.

Deliberately simple (single process, single corpus): enough for a demo, and the seam
where a database and object storage would go in production.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from app.models import AnalysisResult, Corpus, InterviewGuide
from app.parsing import ParseError, ensure_unique_ids, parse_guide, parse_transcript
from app.prompts import PROMPT_VERSION

log = logging.getLogger(__name__)


def build_corpus(transcript_files: list[tuple[str, str]], guide_text: str | InterviewGuide) -> Corpus:
    """Parse ``(filename, text)`` pairs into a corpus. Raises ``ParseError`` with a user-fixable message."""
    if not transcript_files:
        raise ParseError("At least one transcript is required.")
    guide = guide_text if isinstance(guide_text, InterviewGuide) else parse_guide(guide_text)
    ordered = sorted(transcript_files, key=lambda item: item[0].lower())
    transcripts = [parse_transcript(text, name, position) for position, (name, text) in enumerate(ordered, start=1)]
    ensure_unique_ids(transcripts)
    transcripts.sort(key=lambda t: int(t.id[1:]))
    return Corpus(guide=guide, transcripts=transcripts)


def load_sample(sample_dir: Path) -> Corpus:
    files = [(p.name, p.read_text(encoding="utf-8-sig")) for p in sorted(sample_dir.glob("Transcript_*.txt"))]
    guide = (sample_dir / "Interview_Guide.txt").read_text(encoding="utf-8-sig")
    return build_corpus(files, guide)


def corpus_fingerprint(corpus: Corpus) -> str:
    return hashlib.sha256(corpus.model_dump_json().encode("utf-8")).hexdigest()[:16]


def analysis_key(corpus: Corpus, model: str) -> str:
    material = f"{corpus_fingerprint(corpus)}|{model}|{PROMPT_VERSION}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class AnalysisCache:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, key: str) -> Path:
        return self._dir / f"analysis-{key}.json"

    def get(self, key: str) -> AnalysisResult | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            log.warning("Ignoring unreadable cache file %s", path)
            return None

    def put(self, result: AnalysisResult) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(result.key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache file
