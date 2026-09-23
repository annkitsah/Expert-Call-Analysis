from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Groq provides an OpenAI-compatible Chat Completions API.
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Current Groq production model suitable for structured extraction/tool use.
    groq_model: str = "openai/gpt-oss-120b"
    max_output_tokens: int = 8000
    llm_timeout_s: float = 120.0
    llm_max_concurrency: int = 4

    # Q&A context strategy: below this many estimated tokens the whole corpus is sent;
    # above it the app switches to BM25 retrieval over segments.
    full_context_token_budget: int = 60_000
    retrieval_top_k: int = 24
    retrieval_neighbours: int = 1

    sample_dir: Path = ROOT / "data" / "sample"
    cache_dir: Path = ROOT / ".cache"
    max_upload_bytes: int = 1_000_000
    max_upload_files: int = 50


def get_settings() -> Settings:
    return Settings()
