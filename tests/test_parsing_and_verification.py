from __future__ import annotations

import pytest

from app.models import Corpus, RawEvidence
from app.parsing import ParseError, parse_guide, parse_transcript
from app.verification import Verifier, locate_quote
from tests.fakes import GUIDE, THEMES


# ------------------------------------------------------------------ parsing ---
def test_sample_transcripts_parse(corpus: Corpus) -> None:
    assert [t.id for t in corpus.transcripts] == ["T1", "T2", "T3"]
    assert [t.market for t in corpus.transcripts] == ["France", "Germany", "United Kingdom"]
    for t in corpus.transcripts:
        assert len(t.segments) == 14
        assert sum(s.is_expert for s in t.segments) == 7
        assert t.warnings == []


def test_timestamps_and_speakers_come_from_source(corpus: Corpus) -> None:
    seg = corpus.segment("T1-S04")
    assert seg is not None
    assert (seg.timestamp, seg.seconds, seg.speaker, seg.is_expert) == ("01:20", 80, "Dr. Martin", True)
    interviewer = corpus.segment("T1-S03")
    assert interviewer is not None and not interviewer.is_expert


def test_guide_parses_six_questions(corpus: Corpus) -> None:
    assert [q.id for q in corpus.guide.questions] == [1, 2, 3, 4, 5, 6]
    assert corpus.guide.objective.startswith("Understand hospital adoption")
    assert "3–5 years" in corpus.guide.questions[4].text


def test_multiline_turns_hour_timestamps_and_brackets() -> None:
    text = "Expert 4 - Test Person\nRole: R\nMarket: M\n\n[01:02:03]\nTest: first line\ncontinues here\n\n00:05\nInterviewer: hi\n"
    t = parse_transcript(text, "x.txt", 9)
    assert t.id == "T4"
    assert t.segments[0].timestamp == "01:02:03" and t.segments[0].seconds == 3723
    assert t.segments[0].text == "first line continues here"


def test_colon_inside_continuation_is_not_a_new_speaker() -> None:
    t = parse_transcript("00:01\nExpert: Note: budgets differ\n", "x.txt", 1)
    assert t.segments[0].speaker == "Expert" and t.segments[0].text == "Note: budgets differ"


def test_missing_header_falls_back_with_warning() -> None:
    t = parse_transcript("00:00\nAnna: hello there\n", "Transcript_9_Spain.txt", 3)
    assert t.id == "T3" and t.expert_name == "Transcript 9 Spain" and t.warnings


def test_unattributed_text_is_never_citable() -> None:
    t = parse_transcript("00:00\nno speaker prefix here\n\n00:05\nAnna: real speech\n", "x.txt", 1)
    assert t.segments[0].speaker == "Unknown" and not t.segments[0].is_expert
    assert any("no 'Speaker:' prefix" in w for w in t.warnings)


@pytest.mark.parametrize("bad", ["just text, no timestamps", "", "00:00\nInterviewer: only me\n"])
def test_unusable_transcripts_raise_user_readable_errors(bad: str) -> None:
    with pytest.raises(ParseError):
        parse_transcript(bad, "bad.txt", 1)


def test_guide_without_questions_is_rejected() -> None:
    with pytest.raises(ParseError):
        parse_guide("Title\n\nno numbered items")


# ------------------------------------------------------------- verification ---
SEG = "The biggest issue is still capital budget approval. Hospitals may like it, but it isn't easy."


@pytest.mark.parametrize(
    "quote",
    [
        "The biggest issue is still capital budget approval.",
        "the biggest issue is still capital budget approval",  # case + trailing period
        '"The biggest   issue is still capital budget approval."',  # wrapping quotes + spacing
        "but it isn\u2019t easy",  # curly apostrophe vs straight in source
    ],
)
def test_locate_accepts_genuine_variants(quote: str) -> None:
    assert locate_quote(SEG, quote) is not None


def test_locate_returns_span_of_original_text() -> None:
    span = locate_quote(SEG, '"the biggest   issue is still capital budget approval."')
    assert span is not None
    assert SEG[span[0] : span[1]] == "The biggest issue is still capital budget approval."


@pytest.mark.parametrize(
    "quote",
    [
        "The biggest issue is capital budget approval.",  # word dropped
        "The biggest issue is still capital budget sign-off.",  # word changed
        "The biggest ... budget approval.",  # ellipsis stitching
    ],
)
def test_locate_rejects_altered_text(quote: str) -> None:
    assert locate_quote(SEG, quote) is None


def test_locate_rejects_mid_word_and_too_short() -> None:
    assert locate_quote(SEG, "biggest issue is still capital budget approva") is None
    assert locate_quote(SEG, "Hospitals may") is None  # < 3 words


def test_verifier_returns_text_and_timestamp_from_transcript_not_model(corpus: Corpus) -> None:
    v = Verifier(corpus)
    ev = v.resolve(RawEvidence(segment_id="T1-S04", quote="the biggest issue is still capital budget approval"))
    assert ev is not None
    assert ev.quote == "The biggest issue is still capital budget approval"  # original casing restored
    assert (ev.timestamp, ev.expert_name, ev.market) == ("01:20", "Dr. Jean Martin", "France")
    seg = corpus.segment("T1-S04")
    assert seg is not None and seg.text[ev.start : ev.end] == ev.quote


def test_verifier_rejects_fabrication_interviewer_and_wrong_transcript(corpus: Corpus) -> None:
    v = Verifier(corpus)
    fabricated = RawEvidence(segment_id="T1-S04", quote="Regulation is the single biggest barrier to adoption")
    interviewer = RawEvidence(segment_id="T1-S03", quote="What is holding adoption back?")
    nonexistent = RawEvidence(segment_id="T9-S99", quote="Adoption is growing but uneven")
    assert v.resolve(fabricated) is None
    assert v.resolve(interviewer) is None
    assert v.resolve(nonexistent) is None
    # real quote, but attributed to the wrong expert's claim
    real_t2 = RawEvidence(segment_id="T2-S04", quote="Cost is the first barrier.")
    assert v.resolve(real_t2, transcript_id="T1") is None
    assert v.stats.quotes_checked == 4 and v.stats.quotes_verified == 0
    assert len(v.warnings) == 4


def test_every_golden_quote_verifies_against_real_transcripts(corpus: Corpus) -> None:
    """Fixture integrity: the hand-written 'model output' must be genuinely verbatim."""
    v = Verifier(corpus)
    for tid, payload in GUIDE.items():
        for a in payload["answers"]:
            for e in a["evidence"]:
                assert v.resolve(RawEvidence(**e), transcript_id=tid) is not None, e
    for group in (THEMES["common_themes"], THEMES["disagreements"]):
        for item in group:
            for p in item["positions"]:
                for e in p["evidence"]:
                    assert v.resolve(RawEvidence(**e), transcript_id=p["expert_id"]) is not None, e
    assert v.stats.quotes_dropped == 0
