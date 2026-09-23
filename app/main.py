"""HTTP API + static frontend.

Run:  uvicorn app.main:create_app --factory --reload
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import qa
from app.analysis import run_analysis
from app.config import Settings, get_settings
from app.llm import LLMClient, LLMError, LLMUnavailableError, GroqLLM
from app.models import AnalysisResult, AskRequest, AskResponse, Corpus, InterviewGuide, Transcript
from app.parsing import ParseError
from app.store import AnalysisCache, analysis_key, build_corpus, corpus_fingerprint, load_sample

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class CorpusResponse(BaseModel):
    fingerprint: str
    analysis_key: str
    guide: InterviewGuide
    transcripts: list[Transcript]


class AnalyzeRequest(BaseModel):
    refresh: bool = False


class Health(BaseModel):
    status: str
    model: str
    llm_configured: bool


class _State:
    def __init__(self, settings: Settings, llm: LLMClient | None) -> None:
        self.settings = settings
        self.llm = llm
        self.corpus: Corpus = load_sample(settings.sample_dir)
        self.cache = AnalysisCache(settings.cache_dir)
        self.analysis_lock = asyncio.Lock()

    def require_llm(self) -> LLMClient:
        if self.llm is None:
            raise LLMUnavailableError(
                "No Groq API key is configured. Set GROQ_API_KEY (see .env.example) and restart."
            )
        return self.llm

    @property
    def model(self) -> str:
        return self.llm.model if self.llm else self.settings.groq_model

    def key(self) -> str:
        return analysis_key(self.corpus, self.model)


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    if llm is None and settings.groq_api_key:
        llm = GroqLLM(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            base_url=settings.groq_base_url,
            max_tokens=settings.max_output_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    state = _State(settings, llm)
    app = FastAPI(title="Expert call analyst", version="1.0.0")

    # ---- error mapping: every failure carries a message the UI can show verbatim ----
    @app.exception_handler(ParseError)
    async def _parse_error(_: Request, exc: ParseError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    # ---- meta ---------------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> Health:
        return Health(status="ok", model=state.model, llm_configured=state.llm is not None)

    # ---- corpus -------------------------------------------------------------------
    def corpus_response() -> CorpusResponse:
        return CorpusResponse(
            fingerprint=corpus_fingerprint(state.corpus),
            analysis_key=state.key(),
            guide=state.corpus.guide,
            transcripts=state.corpus.transcripts,
        )

    @app.get("/api/corpus")
    async def get_corpus() -> CorpusResponse:
        return corpus_response()

    @app.post("/api/corpus/sample")
    async def reset_to_sample() -> CorpusResponse:
        state.corpus = load_sample(settings.sample_dir)
        return corpus_response()

    @app.post("/api/corpus")
    async def upload_corpus(
        transcripts: Annotated[list[UploadFile], File()],
        guide: Annotated[UploadFile | None, File()] = None,
    ) -> CorpusResponse:
        """Replace the transcripts (and optionally the guide). Nothing changes if parsing fails."""
        if len(transcripts) > settings.max_upload_files:
            raise ParseError(f"Too many files: the limit is {settings.max_upload_files}.")

        async def read_text(upload: UploadFile) -> tuple[str, str]:
            name = upload.filename or "upload.txt"
            data = await upload.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise ParseError(f"{name} is larger than {settings.max_upload_bytes // 1000} KB.")
            try:
                return name, data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ParseError(f"{name} is not UTF-8 text.") from exc

        files = [await read_text(u) for u in transcripts]
        guide_input: str | InterviewGuide = state.corpus.guide
        if guide is not None:
            guide_input = (await read_text(guide))[1]
        state.corpus = build_corpus(files, guide_input)
        return corpus_response()

    # ---- analysis -----------------------------------------------------------------
    @app.get("/api/analysis")
    async def get_analysis() -> AnalysisResult | None:
        """Cached analysis for the current corpus/model/prompt, or null. Never calls the LLM."""
        return state.cache.get(state.key())

    @app.post("/api/analysis")
    async def analyze(body: AnalyzeRequest | None = None) -> AnalysisResult:
        llm_client = state.require_llm()
        key = state.key()
        async with state.analysis_lock:  # concurrent clicks share one run instead of paying twice
            if not (body and body.refresh):
                cached = state.cache.get(key)
                if cached is not None:
                    return cached
            result = await run_analysis(
                llm_client, state.corpus, key=key, max_concurrency=settings.llm_max_concurrency
            )
            state.cache.put(result)
            return result

    # ---- Q&A ----------------------------------------------------------------------
    @app.post("/api/ask")
    async def ask(request: AskRequest) -> AskResponse:
        return await qa.ask(state.require_llm(), state.corpus, request, settings)

    # ---- frontend -----------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
