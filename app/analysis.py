"""Interview-guide answers and cross-call themes/disagreements.

Every quote the model proposes goes through ``Verifier``; anything that cannot be
proven verbatim is dropped, and claims left without evidence are demoted or removed
rather than shown as if they were sourced.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any, TypeVar

from app.llm import LLMClient, LLMError, make_tool, structured_validated
from app.models import (
    AnalysisResult,
    Corpus,
    Disagreement,
    ExpertGuideAnswers,
    GuideAnswer,
    Position,
    RawGuideAnswer,
    RawGuideOutput,
    RawPosition,
    RawThemesOutput,
    Theme,
    ThemesResult,
    Transcript,
)
from app.prompts import (
    GUIDE_INSTRUCTIONS,
    THEMES_INSTRUCTIONS,
    guide_user_message,
    render_corpus,
    render_transcript,
    themes_user_message,
)
from app.verification import Verifier

log = logging.getLogger(__name__)
T = TypeVar("T")

GUIDE_TOOL = make_tool(
    "record_guide_answers",
    "Record this expert's answer, coverage and verbatim supporting quotes for every guide question.",
    RawGuideOutput,
)
THEMES_TOOL = make_tool(
    "record_themes",
    "Record common themes and disagreements across the experts, each backed by verbatim quotes.",
    RawThemesOutput,
)


async def answer_guide(
    llm: LLMClient, corpus: Corpus, transcript: Transcript, verifier: Verifier
) -> ExpertGuideAnswers:
    raw = await structured_validated(
        llm,
        RawGuideOutput,
        instructions=GUIDE_INSTRUCTIONS,
        context=render_transcript(transcript),
        messages=[{"role": "user", "content": guide_user_message(corpus.guide)}],
        tool=GUIDE_TOOL,
    )
    by_question: dict[int, RawGuideAnswer] = {}
    for candidate in raw.answers:
        by_question.setdefault(candidate.question_id, candidate)

    answers: list[GuideAnswer] = []
    for question in corpus.guide.questions:
        item = by_question.get(question.id)
        if item is None:
            verifier.warnings.append(
                f"{transcript.expert_name}: the model returned nothing for question {question.id}; re-run the analysis."
            )
            answers.append(
                GuideAnswer(question_id=question.id, coverage="not_addressed", answer="", evidence=[], status="missing")
            )
            continue
        if item.coverage == "not_addressed":
            answers.append(
                GuideAnswer(
                    question_id=question.id,
                    coverage="not_addressed",
                    answer=item.answer.strip(),
                    evidence=[],
                    status="not_addressed",
                )
            )
            continue
        evidence = verifier.resolve_many(item.evidence, transcript_id=transcript.id)
        answer_text = item.answer.strip()
        answers.append(
            GuideAnswer(
                question_id=question.id,
                coverage=item.coverage,
                answer=answer_text,
                evidence=evidence,
                status="verified" if (evidence and answer_text) else "unverified",
            )
        )
    return ExpertGuideAnswers(transcript_id=transcript.id, answers=answers)


def _verified_positions(raw_positions: list[RawPosition], corpus: Corpus, verifier: Verifier) -> list[Position]:
    out: list[Position] = []
    seen: set[str] = set()
    for rp in raw_positions:
        transcript_id = rp.expert_id.strip()
        if corpus.transcript(transcript_id) is None:
            verifier.warnings.append(f"Ignored a position for unknown expert id {transcript_id!r}.")
            continue
        if transcript_id in seen:
            continue
        evidence = verifier.resolve_many(rp.evidence, transcript_id=transcript_id)
        if not evidence:
            verifier.warnings.append(
                f"Dropped {transcript_id}'s position because none of its quotes could be verified."
            )
            continue
        seen.add(transcript_id)
        out.append(Position(transcript_id=transcript_id, position=rp.position.strip(), evidence=evidence))
    return out


async def analyse_themes(llm: LLMClient, corpus: Corpus, verifier: Verifier) -> ThemesResult:
    raw = await structured_validated(
        llm,
        RawThemesOutput,
        instructions=THEMES_INSTRUCTIONS,
        context=render_corpus(corpus),
        messages=[{"role": "user", "content": themes_user_message(corpus.guide)}],
        tool=THEMES_TOOL,
    )

    themes: list[Theme] = []
    for t in raw.common_themes:
        positions = _verified_positions(t.positions, corpus, verifier)
        if len(positions) < 2:
            verifier.warnings.append(f"Dropped theme {t.title!r}: fewer than two experts remained after verification.")
            continue
        themes.append(Theme(title=t.title.strip(), summary=t.summary.strip(), positions=positions))

    disagreements: list[Disagreement] = []
    for d in raw.disagreements:
        positions = _verified_positions(d.positions, corpus, verifier)
        if len(positions) < 2:
            verifier.warnings.append(
                f"Dropped disagreement {d.topic!r}: fewer than two experts remained after verification."
            )
            continue
        disagreements.append(
            Disagreement(
                topic=d.topic.strip(),
                summary=d.summary.strip(),
                strength=d.strength,
                nuance=d.nuance.strip(),
                positions=positions,
            )
        )
    disagreements.sort(key=lambda item: item.strength != "clear")  # stable: clear conflicts first
    return ThemesResult(common_themes=themes, disagreements=disagreements)


async def run_analysis(
    llm: LLMClient, corpus: Corpus, *, key: str, max_concurrency: int
) -> AnalysisResult:
    """One guide call per transcript plus one cross-call call, all in parallel."""
    verifier = Verifier(corpus)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def limited(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    try:
        async with asyncio.TaskGroup() as group:
            guide_tasks = [
                group.create_task(limited(answer_guide(llm, corpus, t, verifier))) for t in corpus.transcripts
            ]
            themes_task = group.create_task(limited(analyse_themes(llm, corpus, verifier)))
    except* LLMError as failures:
        raise failures.exceptions[0] from None

    return AnalysisResult(
        key=key,
        model=llm.model,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        guide_answers=[task.result() for task in guide_tasks],
        themes=themes_task.result(),
        quotes_checked=verifier.stats.quotes_checked,
        quotes_verified=verifier.stats.quotes_verified,
        warnings=verifier.warnings,
    )
