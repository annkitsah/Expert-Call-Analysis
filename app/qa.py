"""Question answering across all transcripts, with verified citations."""

from __future__ import annotations

import re

from app.config import Settings
from app.llm import LLMClient, make_tool, structured_validated
from app.models import (
    AskRequest,
    AskResponse,
    Citation,
    Corpus,
    RawAnswer,
    RawEvidence,
)
from app.prompts import QA_INSTRUCTIONS, render_corpus
from app.retrieval import select_context
from app.verification import Verifier

QA_TOOL = make_tool(
    "record_answer",
    "Record the answer to the analyst's question with numbered, verbatim citations.",
    RawAnswer,
)

_MARKER_RE = re.compile(r"\[(\d+)\]")


async def ask(
    llm: LLMClient,
    corpus: Corpus,
    request: AskRequest,
    settings: Settings,
) -> AskResponse:
    # Retrieval (only used for large corpora) sees the previous user turn too,
    # so that follow-ups like "and in Germany?" still retrieve on-topic segments.
    previous_user = [t.content for t in request.history if t.role == "user"][-1:]

    selection = select_context(
        corpus,
        " ".join([*previous_user, request.question]),
        token_budget=settings.full_context_token_budget,
        top_k=settings.retrieval_top_k,
        neighbours=settings.retrieval_neighbours,
    )

    context = render_corpus(
        corpus,
        None if selection.mode == "full" else selection.segments_by_transcript,
    )

    # Preserve conversation history as actual chat messages so the model can
    # resolve follow-up questions such as "And in Germany?".
    #
    # Previous assistant responses remain conversational context only.
    # They are not treated as evidence; transcript citations are verified
    # independently against the corpus below.
    messages: list[dict[str, str]] = [
        {
            "role": turn.role,
            "content": turn.content,
        }
        for turn in request.history[-6:]
    ]

    messages.append(
        {
            "role": "user",
            "content": request.question,
        }
    )

    raw = await structured_validated(
        llm,
        RawAnswer,
        instructions=QA_INSTRUCTIONS,
        context=context,
        messages=messages,
        tool=QA_TOOL,
    )

    verifier = Verifier(corpus)
    citations: dict[int, Citation] = {}

    for item in sorted(raw.citations, key=lambda c: c.n):
        if item.n in citations:
            continue

        evidence = verifier.resolve(
            RawEvidence(
                segment_id=item.segment_id,
                quote=item.quote,
            )
        )

        if evidence is not None:
            citations[item.n] = Citation(
                n=item.n,
                **evidence.model_dump(),
            )

    # Any marker that does not resolve to a verified citation is shown as [?]
    # rather than silently kept.
    answer = _MARKER_RE.sub(
        lambda m: (
            m.group(0)
            if int(m.group(1)) in citations
            else "[?]"
        ),
        raw.answer,
    ).strip()

    warnings = list(verifier.warnings)

    if not raw.found:
        status = "not_found"
    elif citations:
        status = "verified"
    else:
        status = "unverified"
        warnings.append(
            "No citation in this answer could be verified against the transcripts."
        )

    return AskResponse(
        answer=answer,
        status=status,
        citations=[citations[n] for n in sorted(citations)],
        warnings=warnings,
        context_mode=selection.mode,
        segments_in_context=selection.segment_count,
    )