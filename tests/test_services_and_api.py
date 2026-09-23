from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.analysis import GUIDE_TOOL, THEMES_TOOL, run_analysis
from app.config import ROOT, Settings
from app.llm import LLMError, structured_validated, tool_schema
from app.main import create_app
from app.models import AskRequest, Corpus, RawGuideOutput, RawThemesOutput
from app.qa import QA_TOOL, ask
from app.retrieval import Bm25Index, select_context
from tests.fakes import ASK_NOT_FOUND, ASK_TRAINING, GUIDE, THEMES, FakeLLM


# ------------------------------------------------------------- analysis ---
async def test_run_analysis_happy_path(corpus: Corpus) -> None:
    llm = FakeLLM()
    result = await run_analysis(llm, corpus, key="k", max_concurrency=4)

    assert len(llm.calls_for("record_guide_answers")) == 3 and len(llm.calls_for("record_themes")) == 1
    assert result.quotes_checked == result.quotes_verified > 0
    assert result.warnings == []
    assert [g.transcript_id for g in result.guide_answers] == ["T1", "T2", "T3"]
    for expert in result.guide_answers:
        assert [a.question_id for a in expert.answers] == [1, 2, 3, 4, 5, 6]
        assert all(a.status == "verified" for a in expert.answers)

    q6 = result.guide_answers[0].answers[5].evidence[0]
    assert (q6.timestamp, q6.segment_id, q6.expert_name) == ("06:08", "T1-S14", "Dr. Jean Martin")
    # 'clear' disagreements are ordered before 'partial' ones
    assert [d.strength for d in result.themes.disagreements] == ["clear", "partial"]


async def test_hallucinated_quote_is_dropped_and_answer_demoted(corpus: Corpus) -> None:
    guide = copy.deepcopy(GUIDE)
    # Q3 for T1: replace the real quote with an invented one -> nothing verifiable is left.
    guide["T1"]["answers"][2]["evidence"] = [
        {"segment_id": "T1-S06", "quote": "Reimbursement rates from national insurers are the deciding factor"}
    ]
    # Q1 for T2: keep one real and add one invented quote -> answer stays verified with the real one only.
    guide["T2"]["answers"][0]["evidence"].append({"segment_id": "T2-S02", "quote": "Adoption doubled last year across Germany"})

    result = await run_analysis(FakeLLM(guide=guide), corpus, key="k", max_concurrency=2)

    t1_q3 = result.guide_answers[0].answers[2]
    assert t1_q3.status == "unverified" and t1_q3.evidence == []
    t2_q1 = result.guide_answers[1].answers[0]
    assert t2_q1.status == "verified" and len(t2_q1.evidence) == 1
    assert result.quotes_checked - result.quotes_verified == 2
    assert len([w for w in result.warnings if "Dropped a quote" in w]) == 2


async def test_evidence_from_another_experts_transcript_is_rejected(corpus: Corpus) -> None:
    guide = copy.deepcopy(GUIDE)
    guide["T1"]["answers"][5]["evidence"] = [{"segment_id": "T2-S14", "quote": "Nine to eighteen months is common."}]
    result = await run_analysis(FakeLLM(guide=guide), corpus, key="k", max_concurrency=2)
    assert result.guide_answers[0].answers[5].status == "unverified"


async def test_missing_and_not_addressed_answers(corpus: Corpus) -> None:
    guide = copy.deepcopy(GUIDE)
    del guide["T3"]["answers"][5]  # model skipped question 6
    guide["T2"]["answers"][3] = {"question_id": 4, "coverage": "not_addressed", "answer": "", "evidence": [
        {"segment_id": "T2-S08", "quote": "Very important operationally."}]}  # stray evidence must be ignored
    result = await run_analysis(FakeLLM(guide=guide), corpus, key="k", max_concurrency=2)
    assert result.guide_answers[2].answers[5].status == "missing"
    assert any("question 6" in w for w in result.warnings)
    na = result.guide_answers[1].answers[3]
    assert na.status == "not_addressed" and na.evidence == []


async def test_theme_with_unverifiable_positions_is_removed(corpus: Corpus) -> None:
    themes = copy.deepcopy(THEMES)
    for pos in themes["common_themes"][0]["positions"][1:]:  # leave only 1 verifiable expert
        pos["evidence"] = [{"segment_id": pos["evidence"][0]["segment_id"], "quote": "Made up statement about budgets"}]
    themes["disagreements"][0]["positions"].append({"expert_id": "T7", "position": "ghost", "evidence": []})
    result = await run_analysis(FakeLLM(themes=themes), corpus, key="k", max_concurrency=2)
    assert len(result.themes.common_themes) == 1
    assert any("fewer than two experts" in w for w in result.warnings)
    assert any("unknown expert id 'T7'" in w for w in result.warnings)


async def test_llm_failure_propagates_as_llm_error(corpus: Corpus) -> None:
    class Broken(FakeLLM):
        async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
            raise LLMError("boom")

    with pytest.raises(LLMError, match="boom"):
        await run_analysis(Broken(), corpus, key="k", max_concurrency=2)


async def test_malformed_output_is_retried_once_then_fails() -> None:
    class Flaky(FakeLLM):
        n = 0

        async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
            Flaky.n += 1
            return {"wrong": "shape"} if Flaky.n == 1 else copy.deepcopy(GUIDE["T1"])

    out = await structured_validated(Flaky(), RawGuideOutput, instructions="", context="", messages=[], tool=GUIDE_TOOL)
    assert Flaky.n == 2 and len(out.answers) == 6

    class AlwaysBad(FakeLLM):
        async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
            return {}

    with pytest.raises(LLMError, match="malformed"):
        await structured_validated(AlwaysBad(), RawGuideOutput, instructions="", context="", messages=[], tool=GUIDE_TOOL)


def test_tool_schemas_are_self_contained() -> None:
    for tool in (GUIDE_TOOL, THEMES_TOOL, QA_TOOL):
        assert "$ref" not in str(tool.schema) and "$defs" not in tool.schema
        assert tool.schema["type"] == "object"
    # a *property* named "title" must survive the cosmetic-title stripping
    theme_props = tool_schema(RawThemesOutput)["properties"]["common_themes"]["items"]["properties"]
    assert "title" in theme_props and "summary" in theme_props


async def test_guide_prompt_gives_model_citable_segment_ids(corpus: Corpus) -> None:
    llm = FakeLLM()
    await run_analysis(llm, corpus, key="k", max_concurrency=1)
    contexts = {c["context"] for c in llm.calls_for("record_guide_answers")}
    assert len(contexts) == 3  # one transcript per call, never mixed
    t1 = next(c for c in contexts if 'id="T1"' in c)
    assert "[T1-S02 00:18] Dr. Martin:" in t1 and "T2-S" not in t1
    themes_ctx = llm.calls_for("record_themes")[0]["context"]
    assert all(f'<transcript id="T{i}"' in themes_ctx for i in (1, 2, 3))


# ------------------------------------------------------------------- Q&A ---
async def test_ask_verifies_citations_and_keeps_markers(corpus: Corpus, settings: Settings) -> None:
    resp = await ask(FakeLLM(), corpus, AskRequest(question="What did experts say about training?"), settings)
    assert resp.status == "verified" and resp.context_mode == "full"
    assert [c.n for c in resp.citations] == [1, 2]
    assert (resp.citations[0].timestamp, resp.citations[1].timestamp) == ("03:10", "03:05")
    assert "[1]" in resp.answer and "[2]" in resp.answer and "[?]" not in resp.answer


async def test_ask_replaces_unverifiable_markers(corpus: Corpus, settings: Settings) -> None:
    bad = copy.deepcopy(ASK_TRAINING)
    bad["citations"][1]["quote"] = "training is the top priority for every hospital"
    resp = await ask(FakeLLM(ask=bad), corpus, AskRequest(question="q"), settings)
    assert resp.status == "verified" and [c.n for c in resp.citations] == [1]
    assert "[?]" in resp.answer and "[2]" not in resp.answer
    assert resp.warnings


async def test_ask_all_citations_fail_is_flagged_unverified(corpus: Corpus, settings: Settings) -> None:
    bad = {"found": True, "answer": "Experts love it [1].", "citations": [{"n": 1, "segment_id": "T1-S02", "quote": "Everyone loves robotic surgery so much"}]}
    resp = await ask(FakeLLM(ask=bad), corpus, AskRequest(question="q"), settings)
    assert resp.status == "unverified" and resp.citations == [] and "[?]" in resp.answer


async def test_ask_not_found_path(corpus: Corpus, settings: Settings) -> None:
    resp = await ask(FakeLLM(ask=ASK_NOT_FOUND), corpus, AskRequest(question="What about reimbursement?"), settings)
    assert resp.status == "not_found" and resp.citations == []


async def test_ask_sends_history_and_full_corpus(corpus: Corpus, settings: Settings) -> None:
    llm = FakeLLM()
    req = AskRequest(
        question="And in Germany?",
        history=[{"role": "user", "content": "Timelines?"}, {"role": "assistant", "content": "France: 6-12 months [1]."}],
    )
    await ask(llm, corpus, req, settings)
    call = llm.calls[0]
    assert [m["role"] for m in call["messages"]] == ["user", "assistant", "user"]
    assert all(f'<transcript id="T{i}"' in call["context"] for i in (1, 2, 3))


def test_ask_request_validates_history_shape() -> None:
    with pytest.raises(ValueError):
        AskRequest(question="x", history=[{"role": "assistant", "content": "hi"}])
    with pytest.raises(ValueError):
        AskRequest(question="x", history=[{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError):
        AskRequest(question="", history=[])


# ------------------------------------------------------------ retrieval ---
def test_small_corpus_uses_full_context(corpus: Corpus) -> None:
    sel = select_context(corpus, "timeline", token_budget=60_000, top_k=5, neighbours=1)
    assert sel.mode == "full" and sel.segment_count == 42


def test_large_corpus_switches_to_bm25_with_neighbours(corpus: Corpus) -> None:
    sel = select_context(corpus, "how long does the purchase decision take", token_budget=10, top_k=2, neighbours=1)
    assert sel.mode == "retrieval" and 0 < sel.segment_count < 42
    ids = {s.id for segs in sel.segments_by_transcript.values() for s in segs}
    assert "T1-S14" in ids and "T1-S13" in ids  # the timeline answer and the question before it
    assert set(sel.segments_by_transcript) == {"T1", "T2", "T3"}  # every transcript still listed


def test_bm25_ranks_relevant_segment_first(corpus: Corpus) -> None:
    segments = [s for t in corpus.transcripts for s in t.segments]
    top = Bm25Index(segments).top("nine to eighteen months procurement", 1)
    assert top[0][0].id == "T2-S14"
    assert Bm25Index(segments).top("zzzz qqqq", 3) == []


async def test_ask_uses_retrieval_when_over_budget(corpus: Corpus) -> None:
    tiny = Settings(_env_file=None, full_context_token_budget=10, retrieval_top_k=2)  # type: ignore[call-arg]
    llm = FakeLLM()
    resp = await ask(llm, corpus, AskRequest(question="purchase timeline"), tiny)
    assert resp.context_mode == "retrieval" and resp.segments_in_context < 42
    assert llm.calls[0]["context"].count("[T") == resp.segments_in_context


# ------------------------------------------------------------------ API ---
def test_health_and_corpus(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "model": "fake-model", "llm_configured": True}
    body = client.get("/api/corpus").json()
    assert len(body["transcripts"]) == 3 and len(body["guide"]["questions"]) == 6
    assert client.get("/").status_code == 200


def test_analysis_is_cached_and_refreshable(client: TestClient, fake_llm: FakeLLM) -> None:
    assert client.get("/api/analysis").json() is None
    first = client.post("/api/analysis")
    assert first.status_code == 200 and len(fake_llm.calls) == 4
    assert client.get("/api/analysis").json()["key"] == first.json()["key"]
    client.post("/api/analysis")
    assert len(fake_llm.calls) == 4  # served from cache, no new model calls
    client.post("/api/analysis", json={"refresh": True})
    assert len(fake_llm.calls) == 8


def test_new_corpus_invalidates_cached_analysis(client: TestClient, fake_llm: FakeLLM) -> None:
    client.post("/api/analysis")
    files = [("transcripts", ("Transcript_1_X.txt", "Expert 1 - A B\nRole: r\nMarket: m\n\n00:00\nA: hello there friend\n", "text/plain"))]
    assert client.post("/api/corpus", files=files).status_code == 200
    assert client.get("/api/analysis").json() is None


def test_ask_endpoint(client: TestClient) -> None:
    resp = client.post("/api/ask", json={"question": "What about training?", "history": []})
    assert resp.status_code == 200 and resp.json()["status"] == "verified"
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_upload_errors_are_actionable_and_do_not_change_state(client: TestClient) -> None:
    before = client.get("/api/corpus").json()["fingerprint"]
    bad = client.post("/api/corpus", files=[("transcripts", ("bad.txt", "no timestamps at all", "text/plain"))])
    assert bad.status_code == 422 and "timestamp" in bad.json()["detail"]
    binary = client.post("/api/corpus", files=[("transcripts", ("b.txt", b"\xff\xfe\x00bad", "text/plain"))])
    assert binary.status_code == 422 and "UTF-8" in binary.json()["detail"]
    assert client.get("/api/corpus").json()["fingerprint"] == before


def test_upload_size_limit(settings: Settings, fake_llm: FakeLLM) -> None:
    small = settings.model_copy(update={"max_upload_bytes": 100})
    c = TestClient(create_app(small, llm=fake_llm))
    r = c.post("/api/corpus", files=[("transcripts", ("t.txt", "x" * 500, "text/plain"))])
    assert r.status_code == 422 and "larger than" in r.json()["detail"]


def test_reset_to_sample(client: TestClient) -> None:
    files = [("transcripts", ("Transcript_1_X.txt", "00:00\nA: hello there friend\n", "text/plain"))]
    client.post("/api/corpus", files=files)
    assert len(client.get("/api/corpus").json()["transcripts"]) == 1
    assert len(client.post("/api/corpus/sample").json()["transcripts"]) == 3


def test_missing_api_key_gives_503_with_instructions(settings: Settings) -> None:
    c = TestClient(create_app(settings, llm=None))
    assert c.get("/api/health").json()["llm_configured"] is False
    for r in (c.post("/api/analysis"), c.post("/api/ask", json={"question": "hi"})):
        assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]
    assert c.get("/api/analysis").status_code == 200  # reading cache needs no key


def test_llm_errors_map_to_502(settings: Settings) -> None:
    class Down(FakeLLM):
        async def structured(self, **kwargs):  # type: ignore[no-untyped-def]
            raise LLMError("Could not reach the Groq API.")

    c = TestClient(create_app(settings, llm=Down()))
    r = c.post("/api/analysis")
    assert r.status_code == 502 and "Could not reach" in r.json()["detail"]


def test_sample_files_are_untouched() -> None:
    """The case pack is shipped byte-for-byte; the parser, not the data, handles CRLF."""
    raw = (ROOT / "data" / "sample" / "Transcript_1_France.txt").read_bytes()
    assert b"\r\n" in raw
