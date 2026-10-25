#!/usr/bin/env python3
"""
test_e2e_batch.py
=================

End-to-end test for ``mock_langsmith_server.py`` using the **batched ingest**
wire protocol -- i.e. the langsmith SDK default (``Client(auto_batch_tracing=
True)``), where runs are buffered client-side and shipped in bulk to
``POST /runs/batch`` instead of the classic per-run ``POST /runs`` +
``PATCH /runs/{id}``.

This is the mode used by "my client agent does not support non-batch
tracing" style clients: the client is constructed plainly (no
``auto_batch_tracing=False``), so every trace op goes through the SDK's
background tracing queue.

This script:

  1. Launches the mock LangSmith server (``mock_langsmith_server.py``) as a
     subprocess on a local port.
  2. Constructs a ``langsmith.Client`` with the DEFAULT batched tracing mode
     and points it at the mock server via ``tracing_v2_enabled(client=...)``.
  3. Builds a tiny LangGraph ReAct-style agent (``create_react_agent``) with
     two simple tools (a calculator and a fake weather lookup), backed by
     an OpenAI-compatible chat model:

          base_url = http://open-webai.kt53.net/openai
          model    = qwen3-8-27b
          api_key  = sk-024610f905ea4b4d8eb4cb0b34f6e811

  4. Runs the agent on a prompt that requires at least one tool call, then
     ``client.flush()``es the background tracing queue.
  5. Asserts that the mock server received and *parsed* the batched runs:
     the per-request "⇈ batch ingest: N post · M patch op(s)" summary lines,
     the chain/llm/tool START/END panels, no errored runs, and none of the
     stale "BATCHED INGEST IGNORED" warnings.

Run directly:

    python3 test_e2e_batch.py

Exits non-zero (and prints a diagnostic) if the batched trace never reaches
the mock server, is not logged, if the agent errors out, or if no tool call
happened.
"""

from __future__ import annotations

import faulthandler
import re
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_SCRIPT = HERE / "mock_langsmith_server.py"

LLM_BASE_URL = "http://open-webai.kt53.net/openai"
LLM_MODEL = "qwen3-8-27b"
LLM_API_KEY = "sk-024610f905ea4b4d8eb4cb0b34f6e811"


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.25)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.1)
    raise RuntimeError(f"mock server never opened {host}:{port}")


def main() -> int:
    faulthandler.dump_traceback_later(60, file=sys.stderr, exit=False)
    port = find_free_port()
    host = "127.0.0.1"
    endpoint = f"http://{host}:{port}"

    print(f"[test] starting mock LangSmith server on {endpoint} ...")
    # NOTE: the server's stdout is redirected to a *file*, not a pipe we read
    # from later. If we used stdout=subprocess.PIPE here and only drained it
    # at the very end (after the test finishes), the server's rich-formatted
    # output can fill the OS pipe buffer (~64KB) and block the server mid
    # request -- which in turn blocks every HTTP call made against it,
    # producing a silent full deadlock. Writing straight to a file avoids
    # that class of bug entirely.
    server_log_path = HERE / f".mock_server_{port}.log"
    server_log_file = open(server_log_path, "w")
    server_proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "--host", host, "--port", str(port),
         "--max-len", "160", "--width", "120"],
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        wait_for_port(host, port)
        print("[test] mock server is up.")

        # --- Tracing configuration ------------------------------------------
        # NOTE: unlike test_e2e_langgraph.py we do NOT pass
        # auto_batch_tracing=False. The default (True) makes the SDK buffer
        # runs in a background queue and ship them via POST /runs/batch
        # (the mock server's /info advertises use_multipart_endpoint=false,
        # so the SDK picks the JSON batch endpoint, not multipart).
        from langsmith import Client
        from langchain_core.tracers.context import tracing_v2_enabled
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        client = Client(
            api_url=endpoint,
            api_key="sk-this-is-ignored-by-the-mock-server",
        )

        # --- Tools -----------------------------------------------------------
        @tool
        def calculator(expression: str) -> str:
            """Evaluate a basic arithmetic expression, e.g. '23 * 7'."""
            try:
                # Restricted eval: digits/operators only.
                allowed = set("0123456789+-*/(). ")
                if not set(expression) <= allowed:
                    return "error: invalid characters in expression"
                return str(eval(expression, {"__builtins__": {}}, {}))
            except Exception as e:  # noqa: BLE001
                return f"error: {e}"

        @tool
        def get_weather(city: str) -> str:
            """Look up the (fake) current weather for a given city."""
            fake_db = {
                "paris": "15C, light rain",
                "berlin": "12C, cloudy",
                "tokyo": "22C, sunny",
            }
            return fake_db.get(city.strip().lower(), f"No data for '{city}', assume 20C and clear.")

        # --- LLM (OpenAI-compatible endpoint) ---------------------------------
        llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=0,
            timeout=30,
            max_retries=1,
        )
        print("[test] LLM client constructed", flush=True)

        agent = create_react_agent(llm, tools=[calculator, get_weather])
        print("[test] agent graph constructed", flush=True)

        prompt = (
            "What is 23 * 7, and also what's the weather like in Paris right now? "
            "Use the tools available to you to answer both parts."
        )

        print(f"[test] invoking agent with prompt: {prompt!r}", flush=True)
        with tracing_v2_enabled(project_name="mock-e2e-batch", client=client):
            print("[test] tracing context entered, calling agent.invoke ...", flush=True)
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            print("[test] agent.invoke returned", flush=True)

        final_message = result["messages"][-1]
        final_text = getattr(final_message, "content", str(final_message))
        print("[test] agent final answer:")
        print(final_text)

        # Batched tracing is asynchronous: ops sit in the client's background
        # queue until its consumer thread flushes them to POST /runs/batch.
        # flush() blocks until the queue is empty; the small extra sleep
        # covers the consumer's in-flight final POST.
        try:
            client.flush(timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"[test] WARNING: client.flush() raised: {e!r}")
        time.sleep(1.5)

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        server_log_file.close()

    server_log = server_log_path.read_text(errors="replace")
    print("\n[test] ----- mock server log -----")
    print(server_log)
    print("[test] ----- end of mock server log -----\n")
    server_log_path.unlink(missing_ok=True)

    # --- Assertions --------------------------------------------------------
    ok = True

    # 1. The batched ingest endpoint must have been used AND parsed.
    if "batch ingest:" not in server_log:
        print("[test] FAIL: no '⇈ batch ingest:' summary line in server log -- "
              "the batch endpoint was never used/parsed.")
        ok = False

    # 2. The stale warning must be gone.
    if "BATCHED INGEST IGNORED" in server_log:
        print("[test] FAIL: stale 'BATCHED INGEST IGNORED' warning present.")
        ok = False

    # 3. All run types must have been logged (start AND end).
    for expected in ("START [TOOL]", "END [TOOL]", "START [LLM]", "END [LLM]"):
        if expected not in server_log:
            print(f"[test] FAIL: did not find expected marker {expected!r} in server log")
            ok = False

    # 4. No run may have errored.
    if "END (ERROR)" in server_log:
        print("[test] FAIL: at least one run errored (END (ERROR) in server log).")
        ok = False

    # 5. The agent must have produced a real answer (digits present).
    if not re.search(r"\d", final_text or ""):
        print("[test] FAIL: no digits in the agent's final answer -- tool result missing?")
        ok = False

    # Non-fatal: the SDK should prefer /runs/batch over /runs/multipart here
    # (per the mock's /info), but multipart is implemented too, so only warn.
    if "multipart ingest:" in server_log:
        print("[test] WARNING: multipart endpoint was used (expected batch); "
              "traces still logged (non-fatal).")

    if ok:
        print("[test] PASS: mock server observed the batched (runs/batch) "
              "run tree end-to-end.")
        return 0
    print("[test] FAIL: mock server did not observe the expected batched traces.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())