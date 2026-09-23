# Expert Call Analyst

A production-style research assistant for the Hasamex expert-call case. It analyzes three expert transcripts against an interview guide, returns per-expert answers with exact evidence, identifies cross-call themes and disagreements, and supports questions across the transcripts.

## What the app demonstrates

- Answers all interview-guide questions for each expert.
- Extracts **verbatim quotes** and source timestamps.
- Distinguishes full, partial, and not-addressed coverage.
- Finds common themes and disagreements across calls.
- Answers analyst questions across the transcript corpus.
- Verifies every displayed citation against the original transcript before serving it.
- Falls back to **not covered / unverified** instead of inventing evidence.

## Stack

- **Backend:** FastAPI + Pydantic
- **Frontend:** Vanilla JavaScript + CSS; no build step
- **LLM:** Groq API through its OpenAI-compatible Chat Completions endpoint
- **Default model:** `openai/gpt-oss-120b`
- **Retrieval:** full-context for the small case corpus; BM25 for larger Q&A contexts
- **Tests:** pytest + mocked HTTP transport

Groq documents its OpenAI-compatible endpoint at `https://api.groq.com/openai/v1`, and the configured model supports tool use. See the official Groq documentation for current model availability.

## Run locally

### 1. Create the environment

Conda:

```bash
conda env create -f environment.yml
conda activate expert-call-analyst
```

Or with venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Groq

```bash
cp .env.example .env
```

Set:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Create the key in the Groq console. Never commit `.env`.

### 3. Start the app

```bash
uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Click **Analyze 3 calls**.

Analysis results are cached in `.cache/`, keyed by corpus, model, and prompt version.

## Docker

```bash
docker build -t expert-call-analyst .
docker run --rm -p 8000:8000 --env-file .env expert-call-analyst
```

Then open `http://127.0.0.1:8000`.

## Architecture

```text
transcripts + guide
        |
        v
 deterministic parser
        |
        v
timestamped speaker segments
 {id, timestamp, speaker, expert flag, text}
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
 per-expert guide       themes/disagreements       Q&A
 structured call         structured call       full context or BM25
        |                      |                      |
        +----------------------+----------------------+
                               v
                 deterministic citation verifier
                               |
                               v
                  verified evidence + timestamps
                               |
                               v
                            API/UI
```

### Why the LLM does not control citations

The model proposes only a `segment_id` and quote. The verifier then:

1. Finds that segment in the parsed source corpus.
2. Confirms it belongs to the correct transcript and is an expert turn.
3. Performs a contiguous verbatim quote match.
4. Slices the displayed quote from the source text.
5. Takes the timestamp directly from the parsed source segment.

Therefore the model cannot invent a timestamp or make up a displayed quote.

## Hallucination controls

1. Prompts explicitly restrict factual claims to transcript text.
2. Structured tool calls are validated with Pydantic.
3. Quotes are deterministically verified against source segments.
4. Invalid/unmatched citations are removed and counted.
5. Answers with no verified evidence are marked `unverified`.
6. Themes/disagreements require verified evidence from at least two experts.
7. Q&A explicitly treats previous assistant responses as untrusted context; transcript text remains the only factual source.
8. `not_addressed` and `not_found` states prevent silence from becoming an invented answer.

### Important limitation

Quote verification proves that a quote exists in the transcript, but it does not constitute a full claim-level entailment check of every sentence in the generated summary. A future production version could add claim-level entailment/NLI verification over the verified evidence.

## Scaling to 30+ transcripts

The current three-call case intentionally uses the full corpus where it fits the context budget because retrieval would add unnecessary recall risk.

For a larger corpus:

```text
30+ transcripts
      |
      v
metadata + segment store
      |
      +--> metadata filters
      |
      +--> BM25 lexical retrieval
      |
      +--> embeddings / semantic retrieval
                    |
                    v
                 reranker
                    |
                    v
             top evidence segments
                    |
                    v
              grounded synthesis
```

Guide analysis can remain one structured pass per transcript, followed by map/reduce over verified answers. Production storage would move from local files/in-memory state to object storage + database/search index, with background jobs for long-running analysis.

## Testing

Run:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check app tests
mypy app
```

The LLM tests use a mocked HTTP transport, so they do not require a real Groq key.

## Repository structure

```text
expert-call-analyst/
├── app/
│   ├── analysis.py
│   ├── config.py
│   ├── llm.py
│   ├── main.py
│   ├── models.py
│   ├── parsing.py
│   ├── prompts.py
│   ├── qa.py
│   ├── retrieval.py
│   ├── store.py
│   ├── verification.py
│   └── static/
├── data/sample/
├── tests/
├── Dockerfile
├── environment.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

## Submission notes

Do not commit the real `.env` or API key. The supplied sample transcripts are kept under `data/sample/`. The app can also accept uploaded UTF-8 transcript files and an optional interview-guide file through the API.


### Groq model configuration

The application uses `openai/gpt-oss-120b` through Groq's OpenAI-compatible Chat Completions API. The client uses `max_completion_tokens` and `reasoning_effort="low"` for predictable structured extraction latency. The model supports local function/tool calling and JSON/structured output on Groq.
