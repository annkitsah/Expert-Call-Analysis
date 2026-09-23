"""Deterministic parsing of transcripts and interview guides.

Timestamps and speakers are extracted here, with plain code, so that no citation
metadata ever depends on the LLM.

Expected transcript layout (CRLF or LF)::

    Expert 1 - Dr. Jean Martin        <- optional header: name
    Role: Head of Urology             <- optional
    Market: France                    <- optional

    00:18                             <- timestamp line (MM:SS or HH:MM:SS, optional [ ])
    Dr. Martin: Adoption is growing   <- "Speaker: text" (text may continue on later lines)
"""

from __future__ import annotations

import re
from pathlib import PurePath

from app.models import GuideQuestion, InterviewGuide, Segment, Transcript

_HEADER_RE = re.compile(r"^Expert\s+(?P<num>\d+)\s*[-\u2013\u2014:]\s*(?P<name>.+)$", re.IGNORECASE)
_TS_RE = re.compile(r"^\[?(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})\]?$")
_SPEAKER_RE = re.compile(r"^(?P<speaker>[^:\n]{1,60}):\s*(?P<text>.*)$")
_QUESTION_RE = re.compile(r"^\s*(?P<num>\d+)[.)]\s+(?P<text>\S.*)$")
_INTERVIEWER_PREFIXES = ("interviewer", "moderator", "host")


class ParseError(ValueError):
    """Raised for input the user can fix; the message is shown verbatim."""


def _seconds(match: re.Match[str]) -> int:
    hours = int(match.group("h") or 0)
    return hours * 3600 + int(match.group("m")) * 60 + int(match.group("s"))


def _is_interviewer(speaker: str) -> bool:
    return speaker.strip().lower().startswith(_INTERVIEWER_PREFIXES)


def parse_transcript(text: str, filename: str, fallback_number: int) -> Transcript:
    """Parse one transcript file into timestamped segments."""
    lines = [ln.strip() for ln in text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    number: int | None = None
    name = ""
    role = ""
    market = ""
    warnings: list[str] = []

    # --- header: everything before the first timestamp line -----------------
    first_ts = next((i for i, ln in enumerate(lines) if _TS_RE.match(ln)), None)
    if first_ts is None:
        raise ParseError(f"{filename}: no timestamp lines (e.g. '00:18') found, so nothing can be cited.")
    for ln in lines[:first_ts]:
        if not ln:
            continue
        if m := _HEADER_RE.match(ln):
            number, name = int(m.group("num")), m.group("name").strip()
        elif ln.lower().startswith("role:"):
            role = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("market:"):
            market = ln.split(":", 1)[1].strip()

    if number is None:
        number = fallback_number
    if not name:
        name = PurePath(filename).stem.replace("_", " ")
        warnings.append("No 'Expert N - Name' header found; expert name was taken from the file name.")
    transcript_id = f"T{number}"

    # --- body: (timestamp, speaker line, continuation lines) -----------------
    segments: list[Segment] = []
    i = first_ts
    while i < len(lines):
        ts_match = _TS_RE.match(lines[i])
        if not ts_match:
            i += 1
            continue
        timestamp = lines[i].strip("[]")
        seconds = _seconds(ts_match)
        i += 1
        while i < len(lines) and not lines[i]:
            i += 1
        body: list[str] = []
        while i < len(lines) and lines[i] and not _TS_RE.match(lines[i]):
            body.append(lines[i])
            i += 1
        if not body:
            warnings.append(f"Timestamp {timestamp} has no text after it and was skipped.")
            continue

        speaker_match = _SPEAKER_RE.match(body[0])
        if speaker_match:
            speaker = speaker_match.group("speaker").strip()
            parts = [speaker_match.group("text"), *body[1:]]
            is_expert = not _is_interviewer(speaker)
        else:
            speaker = "Unknown"
            parts = body
            is_expert = False  # never treat unattributed text as expert testimony
            warnings.append(f"Timestamp {timestamp}: no 'Speaker:' prefix; text can be read but not cited.")

        index = len(segments) + 1
        segments.append(
            Segment(
                id=f"{transcript_id}-S{index:02d}",
                transcript_id=transcript_id,
                index=index,
                timestamp=timestamp,
                seconds=seconds,
                speaker=speaker,
                is_expert=is_expert,
                text=" ".join(p.strip() for p in parts if p.strip()),
            )
        )

    if not segments:
        raise ParseError(f"{filename}: no speaker turns could be parsed.")
    if not any(s.is_expert for s in segments):
        raise ParseError(f"{filename}: only interviewer turns were found; there is nothing to analyse.")

    return Transcript(
        id=transcript_id,
        filename=filename,
        expert_name=name,
        role=role,
        market=market,
        segments=segments,
        warnings=warnings,
    )


def parse_guide(text: str) -> InterviewGuide:
    """Parse the interview guide: title, optional objective, numbered questions."""
    lines = [ln.rstrip() for ln in text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        raise ParseError("Interview guide is empty.")
    title = non_empty[0].strip()

    objective = ""
    for idx, ln in enumerate(lines):
        if ln.lower().startswith("project objective:"):
            chunk = [ln.split(":", 1)[1].strip()]
            for follow in lines[idx + 1 :]:
                if not follow.strip():
                    break
                chunk.append(follow.strip())
            objective = " ".join(c for c in chunk if c)
            break

    questions = [
        GuideQuestion(id=int(m.group("num")), text=m.group("text").strip())
        for ln in lines
        if (m := _QUESTION_RE.match(ln))
    ]
    if not questions:
        raise ParseError("No numbered questions (e.g. '1. How would you ...') found in the interview guide.")
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ParseError("Interview guide has duplicate question numbers.")
    return InterviewGuide(title=title, objective=objective, questions=questions)


def ensure_unique_ids(transcripts: list[Transcript]) -> None:
    """Guard against two files claiming the same expert number."""
    seen: set[str] = set()
    for t in transcripts:
        if t.id in seen:
            raise ParseError(
                f"{t.filename}: expert number {t.id[1:]} is used by more than one file. "
                "Give each file a distinct 'Expert N' header."
            )
        seen.add(t.id)
