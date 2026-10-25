#!/usr/bin/env python3
"""
test_e2e_multiagent.py
======================

A more demanding E2E test for ``mock_langsmith_server.py``: a toy
multi-agent orchestration traced end-to-end against the local mock
LangSmith server.

Topology
--------

    orchestrator (primary agent, create_react_agent)
      ├── tool: math_specialist  (subagent, create_react_agent)
      │     ├── tool: calculator
      │     └── tool: convert_currency
      └── tool: info_specialist  (subagent, create_react_agent)
            ├── tool: get_weather
            └── tool: city_facts

The orchestrator's system prompt forbids it from answering or computing
anything itself, so it must immediately delegate every subtask to one of
the two specialist subagents (ideally in parallel). Each specialist then
uses one or two tool calls and answers quickly, and the orchestrator
combines the results into a final answer.

This produces a realistic nested run tree with all the message
types/roles you'd expect from a real multi-agent session:

  * chain runs  (both agent graphs + internal graph steps)
  * llm runs    (system / human / ai(+tool_calls) / tool message history)
  * tool runs   (subagent delegation tools + leaf utility tools)

Wire setup is identical to ``test_e2e_langgraph.py``:

  * mock server spawned as a subprocess on a free localhost port
    (stdout redirected to a *file*, never an undrained pipe)
  * LANGCHAIN_ENDPOINT / LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY env vars
  * explicit ``Client(auto_batch_tracing=False)`` so the SDK uses the
    classic per-run POST /runs + PATCH /runs/{id} protocol

Run directly:

    python3 test_e2e_multiagent.py
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


def final_text(result: dict) -> str:
    """Extract the final answer text from an agent graph result."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            joined = "".join(parts).strip()
            if joined:
                return joined
    return ""


def main() -> int:
    # Multi-agent round trips take a while (~8 LLM calls); arm a stack
    # dumper so a silent hang leaves a diagnosis behind (no ptrace here).
    faulthandler.dump_traceback_later(180, file=sys.stderr, exit=False)

    port = find_free_port()
    host = "127.0.0.1"
    endpoint = f"http://{host}:{port}"

    print(f"[test] starting mock LangSmith server on {endpoint} ...")
    # CRITICAL: server stdout goes to a FILE, not a pipe. Rich output from
    # a multi-agent run easily exceeds the ~64KB OS pipe buffer; an
    # undrained PIPE would deadlock the server (and thus the SDK) mid-run.
    server_log_path = HERE / f".mock_server_multiagent_{port}.log"
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

        # --- LangSmith tracing env vars (as required by the task) --------
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_API_KEY"] = "sk-this-is-ignored-by-the-mock-server"
        os.environ["LANGCHAIN_PROJECT"] = "mock-multiagent-test"

        # Import *after* setting env vars so the tracer picks them up.
        from langsmith import Client
        from langchain_core.tracers.context import tracing_v2_enabled
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        # Non-batched client => classic synchronous POST /runs and
        # PATCH /runs/{id} per run, which is what the mock implements.
        client = Client(
            api_url=endpoint,
            api_key="sk-this-is-ignored-by-the-mock-server",
            auto_batch_tracing=False,
        )

        # ------------------------------------------------------------------
        # Leaf tools for the specialists
        # ------------------------------------------------------------------
        @tool
        def calculator(expression: str) -> str:
            """Evaluate a basic arithmetic expression, e.g. '120 * 3 * 1.13'."""
            try:
                allowed = set("0123456789+-*/(). ")
                if not set(expression) <= allowed:
                    return "error: invalid characters in expression"
                return str(eval(expression, {"__builtins__": {}}, {}))
            except Exception as e:  # noqa: BLE001
                return f"error: {e}"

        @tool
        def convert_currency(amount: float, to_currency: str) -> str:
            """Convert a USD amount into another currency (fake fixed rates: EUR, GBP, JPY)."""
            rates = {"EUR": 0.92, "GBP": 0.79, "JPY": 151.0, "USD": 1.0}
            rate = rates.get(to_currency.strip().upper())
            if rate is None:
                return f"error: no rate for {to_currency!r} (have: {sorted(rates)})"
            return f"{amount * rate:.2f} {to_currency.strip().upper()}"

        @tool
        def get_weather(city: str) -> str:
            """Look up the (fake) current weather for a given city."""
            fake_db = {
                "paris": "15C, light rain",
                "berlin": "12C, cloudy",
                "tokyo": "22C, sunny",
            }
            return fake_db.get(city.strip().lower(), f"No data for '{city}', assume 20C and clear.")

        @tool
        def city_facts(city: str) -> str:
            """Look up basic facts (country, population) for a given city."""
            fake_db = {
                "paris": ("France", 2_102_650),
                "berlin": ("Germany", 3_664_088),
                "tokyo": ("Japan", 13_960_236),
            }
            entry = fake_db.get(city.strip().lower())
            if entry is None:
                return f"No data for '{city}'."
            country, population = entry
            return f"{city.title()} is in {country} and has a population of {population}."

        # ------------------------------------------------------------------
        # LLM (OpenAI-compatible remote endpoint)
        # ------------------------------------------------------------------
        llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=0,
            timeout=45,
            max_retries=1,
        )
        print("[test] LLM client constructed", flush=True)

        # ------------------------------------------------------------------
        # Subagents (each a full ReAct agent with its own tools + prompt)
        # ------------------------------------------------------------------
        math_specialist = create_react_agent(
            llm,
            tools=[calculator, convert_currency],
            name="math_specialist",
            prompt=(
                "You are math_specialist, an arithmetic and currency-conversion expert. "
                "You MUST use your calculator tool for ALL arithmetic and your "
                "convert_currency tool for ALL currency conversions -- never compute "
                "anything in your head. Keep your final answer to one short sentence."
            ),
        )

        info_specialist = create_react_agent(
            llm,
            tools=[get_weather, city_facts],
            name="info_specialist",
            prompt=(
                "You are info_specialist, a city-information expert. "
                "You MUST use your get_weather and city_facts tools for ALL lookups -- "
                "never answer from memory. Keep your final answer to one short sentence."
            ),
        )
        print("[test] subagents constructed", flush=True)

        # ------------------------------------------------------------------
        # Subagents wrapped as tools for the primary agent
        # ------------------------------------------------------------------
        @tool
        def math_specialist_task(task: str) -> str:
            """Delegate an arithmetic / currency-conversion subtask to the math specialist."""
            result = math_specialist.invoke({"messages": [{"role": "user", "content": task}]})
            return final_text(result) or "(math_specialist returned no text)"

        @tool
        def info_specialist_task(task: str) -> str:
            """Delegate a city-lookup subtask (weather, facts) to the info specialist."""
            result = info_specialist.invoke({"messages": [{"role": "user", "content": task}]})
            return final_text(result) or "(info_specialist returned no text)"

        # ------------------------------------------------------------------
        # Primary agent: orchestrator with quick-delegation policy
        # ------------------------------------------------------------------
        orchestrator = create_react_agent(
            llm,
            tools=[math_specialist_task, info_specialist_task],
            name="orchestrator",
            prompt=(
                "You are the orchestrator of a small team of specialist agents. "
                "HARD RULES:\n"
                "1. You MUST NOT answer factual questions yourself and MUST NOT do any "
                "arithmetic yourself -- you have no such ability.\n"
                "2. IMMEDIATELY delegate every subtask to the matching specialist tool "
                "(math_specialist_task for math/conversions, info_specialist_task for "
                "city weather/facts). Call multiple tools in the SAME turn when the "
                "request has multiple independent parts.\n"
                "3. Do not ask the user questions; just delegate.\n"
                "4. Once all specialists have answered, combine their results into one "
                "concise final answer."
            ),
        )
        print("[test] orchestrator constructed", flush=True)

        task_prompt = (
            "Quick trip sanity check, answer all parts: "
            "(1) my Paris hotel is 120 per night for 3 nights plus 13% tax -- what is the total? "
            "(2) convert that total to EUR. "
            "(3) what is the weather in Paris right now? "
            "(4) what is the population of Berlin?"
        )

        print(f"[test] invoking orchestrator with task: {task_prompt!r}", flush=True)
        with tracing_v2_enabled(project_name="mock-multiagent-test", client=client):
            print("[test] tracing context entered, calling orchestrator.invoke ...", flush=True)
            result = orchestrator.invoke({"messages": [{"role": "user", "content": task_prompt}]})
            print("[test] orchestrator.invoke returned", flush=True)

        answer = final_text(result)
        print("[test] orchestrator final answer:")
        print(answer)

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

    def need(marker: str) -> None:
        nonlocal ok
        if marker not in server_log:
            print(f"[test] FAIL: did not find expected marker {marker!r} in server log")
            ok = False

    for expected in ("START [CHAIN]", "END [CHAIN]", "START [LLM]", "END [LLM]",
                     "START [TOOL]", "END [TOOL]"):
        need(expected)

    # Delegation to both subagents must have happened.
    need("math_specialist")
    need("info_specialist")

    # Leaf tools must have been used (at least 3 of the 4).
    leaf_tools = ("calculator", "convert_currency", "get_weather", "city_facts")
    seen_leafs = sum(1 for t in leaf_tools if t in server_log)
    if seen_leafs < 3:
        print(f"[test] FAIL: expected >=3 distinct leaf tools in log, saw {seen_leafs} "
              f"({[t for t in leaf_tools if t in server_log]})")
        ok = False

    # No run should have errored.
    if "END (ERROR)" in server_log:
        print("[test] FAIL: at least one run errored (see log above)")
        ok = False

    # Final answer should contain digits (hotel total etc.).
    if not any(ch.isdigit() for ch in answer):
        print("[test] FAIL: orchestrator final answer contains no digits")
        ok = False

    if ok:
        print("[test] PASS: mock server observed the full multi-agent run tree end-to-end.")
        return 0
    print("[test] FAIL: multi-agent trace was incomplete or broken.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())