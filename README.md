# mock_langsmith_tracer

Single-file local mock server for the LangSmith tracing API (FastAPI + Uvicorn + Rich).
Point LangChain/LangGraph/LangSmith SDKs at it instead of smith.langchain.com and watch
your agents' traces live in your terminal.

## Run it

    python3 mock_langsmith_server.py --port 1984 --max-len 200 [--width 120]
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_ENDPOINT=http://127.0.0.1:1984
    export LANGCHAIN_API_KEY=anything-goes

Implements the full ingest protocol: classic per-run POST/PATCH /runs, batched
/runs/batch, /runs/multipart, /info and /sessions.

## Views

- `--view full` (default): nested color-coded panels per run -- the whole chain, for debugging.
- `--view timeline`: high-level conversation timeline (agent start/end, human/LLM messages,
  tool calls/results, errors), one labelled line per event, propagation duplicates suppressed.

## Tests

    python3 test_e2e_langgraph.py && python3 test_e2e_multiagent.py
    python3 test_e2e_batch.py && python3 test_e2e_timeline.py

E2E tests run real agents against a real LLM; see AGENTS.md for details.
