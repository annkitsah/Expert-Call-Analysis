"""Domain and API models.

Two families live here, deliberately kept apart:

* ``Raw*``  - the shape the LLM is asked to return. Nothing in these is trusted.
* the rest - what the app stores and serves *after* deterministic verification.
  Timestamps, speakers and quote text in these always come from the parsed
  transcript, never from the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

Coverage = Literal["full", "partial", "not_addressed"]


# --------------------------------------------------------------------------- #
# Source material
# --------------------------------------------------------------------------- #
class Segment(BaseModel):
    """One timestamped speaker turn. The unit of citation."""

    id: str  # e.g. "T1-S05"
    transcript_id: str  # e.g. "T1"
    index: int  # 1-based position within the transcript
    timestamp: str  # exactly as written in the source, e.g. "01:20"
    seconds: int
    speaker: str
    is_expert: bool
    text: str


class Transcript(BaseModel):
    id: str
    filename: str
    expert_name: str
    role: str = ""
    market: str = ""
    segments: list[Segment]
    warnings: list[str] = Field(default_factory=list)


class GuideQuestion(BaseModel):
    id: int
    text: str


class InterviewGuide(BaseModel):
    title: str
    objective: str = ""
    questions: list[GuideQuestion]


class Corpus(BaseModel):
    guide: InterviewGuide
    transcripts: list[Transcript]

    _segments: dict[str, Segment] = PrivateAttr(default_factory=dict)
    _transcripts: dict[str, Transcript] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        self._transcripts = {t.id: t for t in self.transcripts}
        self._segments = {s.id: s for t in self.transcripts for s in t.segments}

    def segment(self, segment_id: str) -> Segment | None:
        return self._segments.get(segment_id)

    def transcript(self, transcript_id: str) -> Transcript | None:
        return self._transcripts.get(transcript_id)


# --------------------------------------------------------------------------- #
# Raw LLM output schemas (untrusted). Field descriptions are sent to the model.
# --------------------------------------------------------------------------- #
class RawEvidence(BaseModel):
    segment_id: str = Field(
        description="Exact segment ID as shown in square brackets in the transcript, e.g. T1-S05."
    )
    quote: str = Field(
        description=(
            "Text copied character-for-character from that ONE segment. Contiguous, 6-40 words, "
            "spoken by the expert. No ellipses, no bracketed edits, no paraphrase."
        )
    )


class RawGuideAnswer(BaseModel):
    question_id: int
    coverage: Coverage = Field(
        description=(
            "full = expert answered the question directly; partial = expert touched on it or "
            "answered only part of it; not_addressed = nothing relevant in the transcript."
        )
    )
    answer: str = Field(
        description="1-3 sentences, based only on what the expert said. Empty string if not_addressed."
    )
    evidence: list[RawEvidence] = Field(
        description="1-3 supporting quotes. Must be empty if coverage is not_addressed."
    )


class RawGuideOutput(BaseModel):
    answers: list[RawGuideAnswer]


class RawPosition(BaseModel):
    expert_id: str = Field(description="Transcript ID of the expert, e.g. T1.")
    position: str = Field(description="One or two sentences stating this expert's view on the topic.")
    evidence: list[RawEvidence] = Field(description="1-2 verbatim supporting quotes from this expert.")


class RawTheme(BaseModel):
    title: str = Field(description="Short noun phrase.")
    summary: str = Field(description="One or two sentences on what the experts share.")
    positions: list[RawPosition] = Field(description="One entry per expert who supports the theme (at least two).")


class RawDisagreement(BaseModel):
    topic: str = Field(description="Short noun phrase.")
    summary: str = Field(description="One or two sentences stating precisely what differs.")
    strength: Literal["clear", "partial"] = Field(
        description=(
            "clear = experts give conflicting answers to the same question; partial = differences of "
            "degree, emphasis or scope."
        )
    )
    nuance: str = Field(
        description=(
            "Caveats on comparability that come from what the experts actually said (e.g. one speaks "
            "about 'some centres', another about 'the whole market'). Empty string if none."
        )
    )
    positions: list[RawPosition] = Field(description="One entry per expert who addressed the topic (at least two).")


class RawThemesOutput(BaseModel):
    common_themes: list[RawTheme]
    disagreements: list[RawDisagreement]


class RawCitation(BaseModel):
    n: int = Field(description="Citation number used as [n] in the answer text. Start at 1.")
    segment_id: str
    quote: str = Field(description="Verbatim, contiguous text from that one expert segment. 4-40 words.")


class RawAnswer(BaseModel):
    found: bool = Field(description="False if the transcripts contain nothing that answers the question.")
    answer: str = Field(description="Plain text. Every factual claim ends with a citation marker like [1].")
    citations: list[RawCitation]


# --------------------------------------------------------------------------- #
# Verified outputs (what the API serves)
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    """A quote that has been proven to appear verbatim in the cited segment."""

    segment_id: str
    transcript_id: str
    expert_name: str
    market: str
    speaker: str
    timestamp: str
    seconds: int
    quote: str  # sliced from the transcript, not echoed from the model
    start: int  # character offsets of ``quote`` within the segment text
    end: int


class GuideAnswer(BaseModel):
    question_id: int
    coverage: Coverage
    answer: str
    evidence: list[Evidence]
    # verified      - has >=1 verified quote
    # unverified    - model claimed an answer but no quote survived verification
    # not_addressed - expert did not discuss it
    # missing       - the model returned nothing for this question
    status: Literal["verified", "unverified", "not_addressed", "missing"]


class ExpertGuideAnswers(BaseModel):
    transcript_id: str
    answers: list[GuideAnswer]


class Position(BaseModel):
    transcript_id: str
    position: str
    evidence: list[Evidence]


class Theme(BaseModel):
    title: str
    summary: str
    positions: list[Position]


class Disagreement(BaseModel):
    topic: str
    summary: str
    strength: Literal["clear", "partial"]
    nuance: str = ""
    positions: list[Position]


class ThemesResult(BaseModel):
    common_themes: list[Theme] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)


class VerificationStats(BaseModel):
    quotes_checked: int = 0
    quotes_verified: int = 0

    @property
    def quotes_dropped(self) -> int:
        return self.quotes_checked - self.quotes_verified


class AnalysisResult(BaseModel):
    key: str  # cache key: corpus + model + prompt version
    model: str
    generated_at: str
    guide_answers: list[ExpertGuideAnswers]
    themes: ThemesResult
    quotes_checked: int
    quotes_verified: int
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Q&A API
# --------------------------------------------------------------------------- #
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _history_alternates(self) -> AskRequest:
        for i, turn in enumerate(self.history):
            expected = "user" if i % 2 == 0 else "assistant"
            if turn.role != expected:
                raise ValueError("history must alternate user/assistant, starting with user")
        if self.history and self.history[-1].role != "assistant":
            raise ValueError("history must end with an assistant turn")
        return self


class Citation(Evidence):
    n: int


class AskResponse(BaseModel):
    answer: str
    status: Literal["verified", "unverified", "not_found"]
    citations: list[Citation]
    warnings: list[str] = Field(default_factory=list)
    context_mode: Literal["full", "retrieval"]
    segments_in_context: int
