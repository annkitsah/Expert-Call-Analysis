"""Test doubles: a fake LLM and hand-written 'golden' model outputs.

The golden outputs use quotes copied from the real sample transcripts, so tests that
feed them through the verifier also prove the verifier accepts genuine model behaviour.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from typing import Any

from app.llm import ToolSpec


def ev(segment_id: str, quote: str) -> dict[str, str]:
    return {"segment_id": segment_id, "quote": quote}


def ans(qid: int, coverage: str, answer: str, *evidence: dict[str, str]) -> dict[str, Any]:
    return {"question_id": qid, "coverage": coverage, "answer": answer, "evidence": list(evidence)}


GUIDE: dict[str, dict[str, Any]] = {
    "T1": {
        "answers": [
            ans(1, "full", "Adoption is growing but concentrated in larger academic hospitals and private centres; smaller regional hospitals are much slower.",
                ev("T1-S02", "Adoption is growing, but it is still concentrated in larger academic hospitals and private centres with stronger capital budgets.")),
            ans(2, "full", "Capital budget approval is the biggest barrier; committees need a strong economic case.",
                ev("T1-S04", "The biggest issue is still capital budget approval."),
                ev("T1-S04", "purchasing committees need a strong economic case before approving a system")),
            ans(3, "full", "ROI is very important; finance wants to see utilisation, volume, maintenance cost and payback.",
                ev("T1-S06", "the finance team wants to understand utilisation, procedure volume, maintenance cost and whether the system will actually pay for itself")),
            ans(4, "full", "Training matters most in year one because utilisation depends on several trained surgeons; clinical outcomes are necessary but not sufficient.",
                ev("T1-S08", "If only one surgeon can use the system, the economics become difficult."),
                ev("T1-S10", "Clinical outcomes are necessary, but they are not enough on their own.")),
            ans(5, "full", "Steady rather than explosive growth; perhaps 15 to 20 percent more procedures annually in some of the stronger centres.",
                ev("T1-S12", "I would expect maybe 15 to 20 percent more procedures annually in some of the stronger centres, but smaller hospitals will remain slower.")),
            ans(6, "full", "Six to twelve months once the hospital is serious, longer if pushed to the next budget cycle.",
                ev("T1-S14", "Six to twelve months is realistic once the hospital becomes serious.")),
        ]
    },
    "T2": {
        "answers": [
            ans(1, "full", "Growing but uneven; large university hospitals are ahead while many smaller hospitals wait.",
                ev("T2-S02", "It is growing, but adoption is quite uneven.")),
            ans(2, "full", "Cost is the first barrier; the second is proving the system will be used enough.",
                ev("T2-S04", "Cost is the first barrier."),
                ev("T2-S04", "The second issue is proving that the system will be used enough.")),
            ans(3, "full", "The economic case decides whether a purchase is approved, even though a clinical case helps.",
                ev("T2-S06", "A strong clinical case helps, but the economic case decides whether it gets approved.")),
            ans(4, "partial", "Training is very important operationally because one comfortable surgeon means poor utilisation; clinical outcomes are mentioned only as helpful.",
                ev("T2-S08", "If the hospital buys a system but only one surgeon is comfortable using it, utilisation will be poor."),
                ev("T2-S06", "A strong clinical case helps")),
            ans(5, "full", "Gradual growth, in the high single digits to low double digits rather than around 20 percent market-wide.",
                ev("T2-S12", "I would expect continued growth, but probably closer to high single digits or low double digits in procedure volumes")),
            ans(6, "full", "Nine to eighteen months is common.",
                ev("T2-S14", "Nine to eighteen months is common.")),
        ]
    },
    "T3": {
        "answers": [
            ans(1, "full", "Increasing; standard for selected procedures in some larger NHS trusts, but access varies by hospital.",
                ev("T3-S02", "Adoption is increasing, and in some larger NHS trusts robotic surgery is becoming standard for selected procedures.")),
            ans(2, "full", "Funding matters, but training capacity is described as just as important.",
                ev("T3-S04", "Funding is important, but I would say training capacity is just as important.")),
            ans(3, "full", "ROI matters but the discussion is not purely financial; economics and clinical strategy are balanced.",
                ev("T3-S06", "It matters, but the discussion is not always purely financial."),
                ev("T3-S08", "I would say economics and clinical strategy are balanced.")),
            ans(4, "full", "Training capacity for surgeons and theatre staff is critical, and hospitals weigh patient outcomes.",
                ev("T3-S04", "if you cannot train enough surgeons and theatre staff, adoption stalls"),
                ev("T3-S06", "Hospitals also consider patient outcomes, length of stay, surgeon recruitment")),
            ans(5, "full", "Quite positive; adoption could accelerate, with growth above 15 percent annually in some areas.",
                ev("T3-S10", "I could see procedure growth above 15 percent annually in some areas.")),
            ans(6, "full", "Around six to nine months if funding is available; much longer if a new capital cycle is needed.",
                ev("T3-S12", "Around six to nine months can happen if funding is already available.")),
        ]
    },
}

THEMES: dict[str, Any] = {
    "common_themes": [
        {
            "title": "Adoption is uneven and concentrated in larger centres",
            "summary": "All three experts describe growth led by large hospitals, with smaller or regional sites behind.",
            "positions": [
                {"expert_id": "T1", "position": "Concentrated in academic hospitals and private centres.",
                 "evidence": [ev("T1-S02", "Smaller regional hospitals are much slower.")]},
                {"expert_id": "T2", "position": "Large university hospitals lead; many smaller ones wait.",
                 "evidence": [ev("T2-S02", "Large university hospitals are much more advanced")]},
                {"expert_id": "T3", "position": "Larger NHS trusts lead; access varies by hospital.",
                 "evidence": [ev("T3-S02", "access still varies significantly by hospital")]},
            ],
        },
        {
            "title": "Trained people and volume make the business case",
            "summary": "Utilisation depends on having enough trained surgeons and procedure volume.",
            "positions": [
                {"expert_id": "T1", "position": "One trained surgeon makes the economics difficult.",
                 "evidence": [ev("T1-S08", "Hospitals want several surgeons trained so utilisation is high enough.")]},
                {"expert_id": "T2", "position": "A single comfortable surgeon leads to poor utilisation.",
                 "evidence": [ev("T2-S08", "That weakens the business case.")]},
                {"expert_id": "T3", "position": "Adoption needs trained people and enough volume.",
                 "evidence": [ev("T3-S14", "Hospitals need enough trained people and enough procedure volume to make the programme sustainable.")]},
            ],
        },
    ],
    "disagreements": [
        {
            "topic": "Growth outlook",
            "summary": "France and the UK cite growth of 15%+ in some settings; Germany expects high single to low double digits.",
            "strength": "partial",
            "nuance": "France and the UK refer to some centres or areas; Germany refers to the whole market.",
            "positions": [
                {"expert_id": "T1", "position": "15 to 20 percent in stronger centres.",
                 "evidence": [ev("T1-S12", "maybe 15 to 20 percent more procedures annually in some of the stronger centres")]},
                {"expert_id": "T2", "position": "High single to low double digits.",
                 "evidence": [ev("T2-S12", "rather than something like 20 percent across the whole market")]},
                {"expert_id": "T3", "position": "Above 15 percent in some areas.",
                 "evidence": [ev("T3-S10", "procedure growth above 15 percent annually in some areas")]},
            ],
        },
        {
            "topic": "How decisive economics are",
            "summary": "France and Germany treat the economic case as decisive; the UK expert says economics and clinical strategy are balanced.",
            "strength": "clear",
            "nuance": "",
            "positions": [
                {"expert_id": "T2", "position": "The economic case decides approval.",
                 "evidence": [ev("T2-S06", "the economic case decides whether it gets approved")]},
                {"expert_id": "T3", "position": "Economics and clinical strategy are balanced.",
                 "evidence": [ev("T3-S08", "I would not say finance alone decides the purchase.")]},
            ],
        },
    ],
}

ASK_TRAINING: dict[str, Any] = {
    "found": True,
    "answer": "Dr. Martin says training matters most in year one [1]. Anna Keller says one comfortable surgeon means poor utilisation [2].",
    "citations": [
        {"n": 1, "segment_id": "T1-S08", "quote": "Training matters, especially in the first year."},
        {"n": 2, "segment_id": "T2-S08", "quote": "only one surgeon is comfortable using it, utilisation will be poor"},
    ],
}

ASK_NOT_FOUND: dict[str, Any] = {
    "found": False,
    "answer": "None of the experts discussed reimbursement.",
    "citations": [],
}


class FakeLLM:
    """Records calls and returns canned tool inputs. Deep-copies so tests can mutate freely."""

    model = "fake-model"

    def __init__(self, *, guide=None, themes=None, ask=None) -> None:  # type: ignore[no-untyped-def]
        self.guide = guide if guide is not None else GUIDE
        self.themes = themes if themes is not None else THEMES
        self.ask: Any = ask if ask is not None else ASK_TRAINING
        self.calls: list[dict[str, Any]] = []

    async def structured(self, *, instructions: str, context: str, messages: list[dict[str, str]], tool: ToolSpec) -> dict[str, Any]:
        self.calls.append({"tool": tool.name, "context": context, "messages": messages, "instructions": instructions})
        if tool.name == "record_guide_answers":
            tid = re.search(r'<transcript id="(T\d+)"', context).group(1)  # type: ignore[union-attr]
            return copy.deepcopy(self.guide[tid])
        if tool.name == "record_themes":
            return copy.deepcopy(self.themes)
        if tool.name == "record_answer":
            responder: Callable[..., Any] | Any = self.ask
            return copy.deepcopy(responder(messages) if callable(responder) else responder)
        raise AssertionError(f"unexpected tool {tool.name}")

    def calls_for(self, tool_name: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["tool"] == tool_name]
