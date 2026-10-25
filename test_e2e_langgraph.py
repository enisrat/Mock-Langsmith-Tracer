#!/usr/bin/env python3
"""
test_e2e_langgraph.py
======================

End-to-end test for ``mock_langsmith_server.py``.

This script:

  1. Launches the mock LangSmith server (``mock_langsmith_server.py``) as a
     subprocess on a local port.
  2. Points the LangSmith/LangChain tracing SDK at that local mock server
     via LANGCHAIN_ENDPOINT / LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY.
  3. Builds a tiny LangGraph ReAct-style agent (``create_react_agent``) with
     two simple tools (a calculator and a fake weather lookup), backed by
     an OpenAI-compatible chat model:

         base_url = http://open-webai.kt53.net/openai
         model    = qwen3-8-27b
         api_key  = sk-024610f905ea4b4d8eb4cb0b34f6e811

  4. Runs the agent on a prompt that requires at least one tool call, which
     causes LangGraph/LangChain to emit a chain/llm/tool run tree that gets
     traced -- i.e. POSTed/PATCHed -- to the mock server.
  5. Asserts that the mock server actually received the expected run_types
     (chain, llm, tool) by parsing its stdout log.

Run directly:

    python3 test_e2e_langgraph.py

Exits non-zero (and prints a diagnostic) if the trace never reaches the
mock server, if the agent errors out, or if no tool call happened.
"""

from __future__ import annotations

import faulthandler
import os
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
    faulthandler.dump_traceback_later(45, file=sys.stderr, exit=False)
    port = find_free_port()
    host = "127.0.0.1"
    endpoint = f"http://{host}:{port}"

    print(f"[test] starting mock LangSmith server on {endpoint} ...")
    # NOTE: the server's stdout is redirected to a *file*, not a pipe we read
    # from later. If we used stdout=subprocess.PIPE here and only drained it
    # at the very end (after the test finishes), the server's rich-formatted
    # output (which can be sizeable once large conversation histories are
    # logged) can fill the OS pipe buffer (~64KB) and block the server mid
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

        # --- Configure LangSmith tracing env vars, as requested --------------
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_API_KEY"] = "sk-this-is-ignored-by-the-mock-server"
        os.environ["LANGCHAIN_PROJECT"] = "mock-e2e-test"

        # Import *after* setting env vars so LangChainTracer picks them up.
        from langsmith import Client
        from langchain_core.tracers.context import tracing_v2_enabled
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        # IMPORTANT: auto_batch_tracing=False makes the SDK synchronously
        # POST /runs and PATCH /runs/{id} per run (the classic REST shape),
        # instead of buffering runs and shipping them via the batch/
        # multipart ingest endpoints. This matches exactly the 4 endpoints
        # implemented by the mock server.
        client = Client(
            api_url=endpoint,
            api_key="sk-this-is-ignored-by-the-mock-server",
            auto_batch_tracing=False,
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
        with tracing_v2_enabled(project_name="mock-e2e-test", client=client):
            print("[test] tracing context entered, calling agent.invoke ...", flush=True)
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            print("[test] agent.invoke returned", flush=True)

        final_message = result["messages"][-1]
        final_text = getattr(final_message, "content", str(final_message))
        print("[test] agent final answer:")
        print(final_text)

        # Give the (synchronous, but still network-bound) trace calls a brief
        # moment to land fully and for the server to flush stdout.
        client.flush()
        time.sleep(0.5)

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
    for expected in ("START [TOOL]", "END [TOOL]", "START [LLM]", "END [LLM]"):
        if expected not in server_log:
            print(f"[test] FAIL: did not find expected marker {expected!r} in server log")
            ok = False

    if "SESSION CREATED" not in server_log and "START [CHAIN]" not in server_log and "START [AGENT]" not in server_log:
        # Not strictly required (session creation is optional / implicit),
        # but at least one run should have been recorded.
        print("[test] WARNING: no session/chain/agent run observed (non-fatal)")

    if ok:
        print("[test] PASS: mock server observed LLM and TOOL run traces end-to-end.")
        return 0
    else:
        print("[test] FAIL: mock server did not observe the expected traces.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
