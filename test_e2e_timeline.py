#!/usr/bin/env python3
"""
test_e2e_timeline.py
====================

E2E test for the ``--view timeline`` mode of ``mock_langsmith_server.py``
(and for the agent/thread attribution added to the full view).

Part A -- real LLM, timeline view
---------------------------------

Runs a multi-agent orchestration -- built with ``langchain.agents.create_agent``,
which stamps every run's metadata with ``lc_agent_name`` -- against the mock
server started with ``--view timeline``:

    orchestrator (checkpointer, thread "thread-e2e-a")
      ├── tool: math_specialist_task  -> math_specialist (calculator)
      └── tool: info_specialist_task  -> info_specialist (get_weather)

Two invokes happen on the SAME thread (conversation continuity). The test then
asserts, from the server's log:

  * agent lifecycle markers: "▶ agent <name>" / "✔ agent <name>", once per
    agent invocation (orchestrator exactly twice, once per turn);
  * every human / ai / tool line is labelled with the owning agent's name
    and the thread id ("agent's name with each message");
  * tool calls ("⚒ ...") and tool results ("tool ... -> result") appear;
  * the full chain is SUPPRESSED: no "START [..]" / "END [..]" panels, no
    batch-ingest summaries;
  * propagated message duplicates are SUPPRESSED: the root user task text
    (which the SDK re-sends inside every nested run's inputs) shows up at
    most twice in the whole timeline log (once per turn that mentions it),
    not once per run that carried it;
  * no errors ("✖") and real answers from the LLM.

Part B -- deterministic replay (no LLM): timeline vs full view
--------------------------------------------------------------

A small, hand-built but wire-accurate op sequence (classic POST /runs +
PATCH /runs/{id}, metadata copied from real captured traffic) is replayed
against two fresh server instances:

  * timeline view: the same human text, propagated through several runs of
    the same agent (root chain -> node chain -> llm inputs), is emitted
    EXACTLY once per agent; the sub-agent (different ``lc_agent_name``)
    still gets its own human line; lifecycle/tool/ai markers all present;
    an errored tool run yields exactly one "✖" line;
  * full view: the propagated copies are all still visible (the debug view
    is unchanged), panels are printed, and each panel's meta line shows
    "agent <name> · thread <id>".

Run directly:

    python3 test_e2e_timeline.py
"""

from __future__ import annotations

import faulthandler
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
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


def start_server(port: int, log_name: str, extra_args: list[str]) -> tuple[subprocess.Popen, Path, object]:
    """Spawn the mock server with stdout redirected to a FILE (never an
    undrained pipe -- see AGENTS.md re: the 64KB pipe-buffer deadlock)."""
    log_path = HERE / f".{log_name}_{port}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "--host", "127.0.0.1",
         "--port", str(port), "--max-len", "160", "--width", "140", *extra_args],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        wait_for_port("127.0.0.1", port)
    except Exception:
        proc.terminate()
        log_file.close()
        raise
    return proc, log_path, log_file


def stop_server(proc: subprocess.Popen, log_file: object, log_path: Path) -> str:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()
    log = log_path.read_text(errors="replace")
    log_path.unlink(missing_ok=True)
    return log


def final_text(result: dict) -> str:
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


# --------------------------------------------------------------------------- #
# Part A: real-LLM multi-agent run against a timeline-view server
# --------------------------------------------------------------------------- #

def part_a() -> bool:
    port = find_free_port()
    endpoint = f"http://127.0.0.1:{port}"

    print(f"[test A] starting mock server (--view timeline) on {endpoint} ...")
    proc, log_path, log_file = start_server(port, "mock_server_timeline", ["--view", "timeline"])

    task1 = ("Quick trip sanity check: (1) what is 23 * 7? "
             "(2) what is the weather in Paris right now?")
    task2 = "And what is the weather in Berlin?"
    answers: list[str] = []

    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_API_KEY"] = "sk-this-is-ignored-by-the-mock-server"
        os.environ["LANGCHAIN_PROJECT"] = "mock-timeline-test"

        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from langchain_core.tracers.context import tracing_v2_enabled
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import MemorySaver
        from langsmith import Client

        # Non-batched client => classic synchronous POST /runs and
        # PATCH /runs/{id} per run (deterministic ordering for the timeline).
        client = Client(
            api_url=endpoint,
            api_key="sk-this-is-ignored-by-the-mock-server",
            auto_batch_tracing=False,
        )

        @tool
        def calculator(expression: str) -> str:
            """Evaluate a basic arithmetic expression, e.g. '23 * 7'."""
            try:
                allowed = set("0123456789+-*/(). ")
                if not set(expression) <= allowed:
                    return "error: invalid characters in expression"
                return str(eval(expression, {"__builtins__": {}}, {}))
            except Exception as e:  # noqa: BLE001
                return f"error: {e}"

        @tool
        def get_weather(city: str) -> str:
            """Look up the (fake) current weather for a given city."""
            fake_db = {"paris": "15C, light rain", "berlin": "12C, cloudy",
                       "tokyo": "22C, sunny"}
            return fake_db.get(city.strip().lower(), f"No data for '{city}'.")

        llm = ChatOpenAI(
            base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY,
            temperature=0, timeout=45, max_retries=1,
        )
        print("[test A] LLM client constructed", flush=True)

        math_specialist = create_agent(
            llm, tools=[calculator], name="math_specialist",
            system_prompt=("You are math_specialist. You MUST use the calculator "
                           "tool for ALL arithmetic. Answer in one short sentence."),
        )
        info_specialist = create_agent(
            llm, tools=[get_weather], name="info_specialist",
            system_prompt=("You are info_specialist. You MUST use the get_weather "
                           "tool for ALL lookups. Answer in one short sentence."),
        )

        @tool
        def math_specialist_task(task: str) -> str:
            """Delegate an arithmetic subtask to the math specialist."""
            r = math_specialist.invoke({"messages": [{"role": "user", "content": task}]})
            return final_text(r) or "(math_specialist returned no text)"

        @tool
        def info_specialist_task(task: str) -> str:
            """Delegate a weather subtask to the info specialist."""
            r = info_specialist.invoke({"messages": [{"role": "user", "content": task}]})
            return final_text(r) or "(info_specialist returned no text)"

        orchestrator = create_agent(
            llm,
            tools=[math_specialist_task, info_specialist_task],
            name="orchestrator",
            system_prompt=(
                "You are the orchestrator of a small team of specialist agents. "
                "HARD RULES:\n"
                "1. You MUST NOT answer factual questions yourself and MUST NOT do "
                "any arithmetic yourself -- you have no such ability.\n"
                "2. IMMEDIATELY delegate every subtask to the matching specialist tool "
                "(math_specialist_task for math, info_specialist_task for weather). "
                "Call multiple tools in the SAME turn when the request has multiple "
                "independent parts.\n"
                "3. Do not ask the user questions; just delegate.\n"
                "4. Once all specialists have answered, combine their results into one "
                "concise final answer."
            ),
            checkpointer=MemorySaver(),
        )
        print("[test A] agents constructed", flush=True)

        with tracing_v2_enabled(project_name="mock-timeline-test", client=client):
            print(f"[test A] turn 1: {task1!r}", flush=True)
            r1 = orchestrator.invoke(
                {"messages": [{"role": "user", "content": task1}]},
                config={"configurable": {"thread_id": "thread-e2e-a"}},
            )
            answers.append(final_text(r1))
            print(f"[test A] answer 1: {answers[-1]}", flush=True)

            print(f"[test A] turn 2 (same thread): {task2!r}", flush=True)
            r2 = orchestrator.invoke(
                {"messages": [{"role": "user", "content": task2}]},
                config={"configurable": {"thread_id": "thread-e2e-a"}},
            )
            answers.append(final_text(r2))
            print(f"[test A] answer 2: {answers[-1]}", flush=True)

            client.flush()
            time.sleep(0.5)
    finally:
        server_log = stop_server(proc, log_file, log_path)

    print("\n[test A] ----- mock server log (timeline view) -----")
    print(server_log)
    print("[test A] ----- end of mock server log -----\n")

    ok = True

    def need(marker: str) -> None:
        nonlocal ok
        if marker not in server_log:
            print(f"[test A] FAIL: expected {marker!r} in timeline log")
            ok = False

    def forbid(marker: str) -> None:
        nonlocal ok
        if marker in server_log:
            print(f"[test A] FAIL: {marker!r} must not appear in timeline log")
            ok = False

    # 1. Agent lifecycle: orchestrator opens/closes exactly once per turn.
    if server_log.count("▶ agent orchestrator") != 2:
        print(f"[test A] FAIL: expected exactly 2 '▶ agent orchestrator' lines, "
              f"got {server_log.count('▶ agent orchestrator')}")
        ok = False
    if server_log.count("✔ agent orchestrator") != 2:
        print(f"[test A] FAIL: expected exactly 2 '✔ agent orchestrator' lines, "
              f"got {server_log.count('✔ agent orchestrator')}")
        ok = False
    need("▶ agent math_specialist")
    need("✔ agent math_specialist")
    need("▶ agent info_specialist")
    need("✔ agent info_specialist")

    # 2. Agent name + thread shown with every message line.
    ts_label = r"\[\d{2}:\d{2}:\d{2}\.\d{3}\]\s+"
    for agent in ("orchestrator", "math_specialist", "info_specialist"):
        for role in ("human", "ai"):
            if not re.search(rf"^{ts_label}{agent}\b.*\b{role}\b", server_log, re.M):
                print(f"[test A] FAIL: no timeline line labelled {agent!r} "
                      f"with a {role!r} message")
                ok = False
    if not re.search(rf"^{ts_label}math_specialist\b.*\btool\b.*calculator", server_log, re.M):
        print("[test A] FAIL: no 'tool calculator(...)' line labelled math_specialist")
        ok = False
    need("⚒")            # tool-call markers from ai generations
    need("thread-e2e-a")  # thread id shown on the label column

    # 3. The full chain is suppressed: no run panels, no transport noise.
    for marker in ("START [", "END [", "⇈ batch ingest", "⇈ multipart ingest",
                   "SESSION CREATED", "earlier message"):
        forbid(marker)

    # 4. Dedupe of propagated artifacts: the root user text is re-sent inside
    #    every nested run's inputs by the SDK (and replayed as history by the
    #    checkpointer on turn 2); in the timeline it must yield exactly ONE
    #    human line (a possible verbatim tool-call arg is the only allowed
    #    extra occurrence).
    ts_lbl = rf"{ts_label}orchestrator\b.*human\s+"
    n_human_lines = len(re.findall(ts_lbl, server_log, re.M))
    if n_human_lines != 2:  # exactly one per turn
        print(f"[test A] FAIL: expected exactly 2 orchestrator human lines "
              f"(one per turn), found {n_human_lines}")
        ok = False
    n_task1 = server_log.count(task1)
    if not 1 <= n_task1 <= 2:
        print(f"[test A] FAIL: root task text should appear 1-2x in the timeline "
              f"log (deduped), found {n_task1}x")
        ok = False
    n_task2 = server_log.count(task2)
    if n_task2 != 1:
        print(f"[test A] FAIL: turn-2 text should appear exactly 1x (deduped), "
              f"found {n_task2}x")
        ok = False

    # 5. No errors; real answers.
    forbid("✖")
    for i, ans in enumerate(answers, 1):
        if not ans or not any(ch.isdigit() for ch in ans):
            print(f"[test A] FAIL: answer {i} missing digits: {ans!r}")
            ok = False

    return ok


# --------------------------------------------------------------------------- #
# Part B: deterministic replay of wire-accurate ops against both views
# --------------------------------------------------------------------------- #

HUMAN_ROOT = "What is the weather in Paris?"
HUMAN_SUB = "Weather in Paris please"
THREAD = "thread-x"
T0 = datetime(2026, 10, 25, 10, 15, 0, tzinfo=timezone.utc)


def _iso(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


def _seg(run_id: str, seconds: float) -> str:
    return f"{(T0 + timedelta(seconds=seconds)):%Y%m%dT%H%M%S%f}Z{run_id}"


def _meta(lc: str, node: str | None = None, step: int | None = None) -> dict:
    m: dict = {"lc_agent_name": lc, "thread_id": THREAD}
    if node:
        m["langgraph_node"] = node
    if step is not None:
        m["langgraph_step"] = step
    return m


def _body(run: dict, op: str, outputs=None, error=None, end=None) -> dict:
    b = {
        "id": run["id"],
        "name": run["name"],
        "run_type": run["type"],
        "session_name": "mock-timeline-test",
        "start_time": _iso(run["t0"]),
        "trace_id": run["trace"],
        "dotted_order": run["dotted"],
        "parent_run_id": run["parent"],
        "extra": {"metadata": _meta(run["lc"], run.get("node"), run.get("step"))},
        "tags": [],
    }
    if op == "post":
        b["inputs"] = run.get("inputs")
        b["outputs"] = {}
        b["error"] = None
        b["end_time"] = None
    else:
        b["outputs"] = outputs
        b["error"] = error
        b["end_time"] = _iso(run.get("t1", run["t0"] + 1))
    return b


def _ai_msg(content: str, tool_calls: list[dict] | None = None) -> dict:
    return {
        "lc": 1, "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "content": content, "type": "ai",
            "tool_calls": tool_calls or [],
            "response_metadata": {"model_name": "qwen3-8-27b",
                                  "finish_reason": "tool_calls" if tool_calls else "stop"},
        },
    }


def _gen(msg: dict) -> dict:
    return {"text": msg["kwargs"]["content"], "generation_info": {},
            "type": "ChatGeneration", "message": msg}


def build_replay_ops() -> list[tuple[str, str, dict]]:
    """(http_method, path, body) ops copying the exact wire shapes captured
    from a real create_agent run (see AGENTS.md). Tree:

      orchestrator (boundary)                    [human HUMAN_ROOT]
        └─ chain model (lc=orchestrator)         [HUMAN_ROOT propagated]
             └─ llm ChatOpenAI                  [HUMAN_ROOT propagated]
                  gen: ai -> ⚒ info_specialist_task(HUMAN_SUB)
        └─ chain tools (lc=orchestrator)
             ├─ tool info_specialist_task        (delegation, args HUMAN_SUB)
             │    └─ info_specialist (boundary)  [human HUMAN_SUB]
             │         ├─ llm gen: ai -> ⚒ get_weather(Paris)
             │         ├─ chain tools ─ tool get_weather -> "15C, light rain"
             │         └─ llm gen: ai "15C, light rain"  [HUMAN_SUB propagated]
             └─ tool calculator -> ERROR (errored run)
    """
    ops: list[tuple[str, str, dict]] = []
    trace = str(uuid.uuid4())
    clock = 0.0

    def mk(name, rtype, parent, dotted, lc, inputs=None, **kw) -> dict:
        nonlocal clock
        clock += 0.05
        rid = str(uuid.uuid4())
        run = {"id": rid, "name": name, "type": rtype, "parent": parent["id"] if parent else None,
               "dotted": (f"{dotted}.{_seg(rid, clock)}" if parent else _seg(rid, clock)),
               "trace": trace, "t0": clock, "lc": lc, "inputs": inputs, **kw}
        return run

    def post(run) -> None:
        ops.append(("POST", "/runs", _body(run, "post")))

    def patch(run, outputs=None, error=None) -> None:
        run["t1"] = run["t0"] + 0.4
        ops.append(("PATCH", f"/runs/{run['id']}", _body(run, "patch", outputs, error)))

    # -- root boundary ----------------------------------------------------
    root = mk("orchestrator", "chain", None, "", "orchestrator",
              inputs={"messages": [{"type": "human", "content": HUMAN_ROOT}]})
    post(root)

    # -- orchestrator model node + llm (HUMAN_ROOT propagates) ------------
    dotted_root = root["dotted"]
    model = mk("model", "chain", root, dotted_root, "orchestrator",
               node="model", step=1,
               inputs={"messages": [{"type": "human", "content": HUMAN_ROOT}]})
    post(model)
    llm = mk("ChatOpenAI", "llm", model, model["dotted"], "orchestrator",
             node="model", step=1,
             inputs={"messages": [
                 {"type": "system", "content": "You are the orchestrator."},
                 {"type": "human", "content": HUMAN_ROOT}]})
    post(llm)
    patch(llm, outputs={"generations": [[_gen(_ai_msg("\n\n", [
        {"name": "info_specialist_task", "args": {"task": HUMAN_SUB},
         "id": "call_1", "type": "tool_call"}]))]],
        "llm_output": {}, "run": None, "type": "LLMResult"})
    patch(model, outputs={"messages": [{"type": "ai", "content": ""}]})

    # -- orchestrator tools node ------------------------------------------
    tools = mk("tools", "chain", root, dotted_root, "orchestrator",
               node="tools", step=2, inputs={"input": []})
    post(tools)

    # delegation tool -> subagent boundary (new lc_agent_name)
    deleg = mk("info_specialist_task", "tool", tools, tools["dotted"],
               "orchestrator", node="tools", step=2,
               inputs={"task": HUMAN_SUB})
    post(deleg)
    sub = mk("info_specialist", "chain", deleg, deleg["dotted"], "info_specialist",
             node="tools", step=2,
             inputs={"messages": [{"type": "human", "content": HUMAN_SUB}]})
    post(sub)
    sub_llm1 = mk("ChatOpenAI", "llm", sub, sub["dotted"], "info_specialist",
                  node="model", step=1,
                  inputs={"messages": [
                      {"type": "system", "content": "You are info_specialist."},
                      {"type": "human", "content": HUMAN_SUB}]})
    post(sub_llm1)
    patch(sub_llm1, outputs={"generations": [[_gen(_ai_msg("\n\n", [
        {"name": "get_weather", "args": {"city": "Paris"},
         "id": "call_2", "type": "tool_call"}]))]],
        "llm_output": {}, "run": None, "type": "LLMResult"})

    # subagent tools node + leaf tool (attributed to info_specialist)
    sub_tools = mk("tools", "chain", sub, sub["dotted"], "info_specialist",
                   node="tools", step=2, inputs={"input": []})
    post(sub_tools)
    sub_gw = mk("get_weather", "tool", sub_tools, sub_tools["dotted"],
                "info_specialist", node="tools", step=2,
                inputs={"city": "Paris"})
    post(sub_gw)
    patch(sub_gw, outputs={"output": {"content": "15C, light rain", "type": "tool",
                                      "name": "get_weather", "tool_call_id": "call_2"}})
    patch(sub_tools, outputs={"messages": []})

    sub_llm2 = mk("ChatOpenAI", "llm", sub, sub["dotted"], "info_specialist",
                  node="model", step=3,
                  inputs={"messages": [
                      {"type": "human", "content": HUMAN_SUB},
                      {"type": "ai", "content": ""},
                      {"type": "tool", "content": "15C, light rain"}]})
    post(sub_llm2)
    patch(sub_llm2, outputs={"generations": [[_gen(_ai_msg("15C, light rain"))]],
        "llm_output": {}, "run": None, "type": "LLMResult"})
    patch(sub, outputs={"messages": [{"type": "ai", "content": "15C, light rain"}]})
    patch(deleg, outputs={"output": "15C, light rain"})

    # an ERRORED leaf tool run under the orchestrator
    calc = mk("calculator", "tool", tools, tools["dotted"], "orchestrator",
              node="tools", step=2, inputs={"expression": "23 * 7"})
    post(calc)
    patch(calc, outputs=None, error="ValueError: bad expression")

    patch(tools, outputs={"messages": []})
    patch(root, outputs={"messages": [{"type": "ai", "content": "15C, light rain"}]})
    return ops


def replay(ops: list[tuple[str, str, dict]], port: int) -> None:
    for method, path, body in ops:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()


def replay_as_batch(ops: list[tuple[str, str, dict]], port: int) -> None:
    """Ship the same ops through the SDK-default batched transport
    (POST /runs/batch: all posts first, then all patches)."""
    posts = [b for m, _, b in ops if m == "POST"]
    patches = [b for m, _, b in ops if m == "PATCH"]
    data = json.dumps({"post": posts, "patch": patches}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/runs/batch", data=data, method="POST",
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5).read()


def part_b() -> bool:
    ops = build_replay_ops()
    ok = True

    # ---- timeline view ---------------------------------------------------
    port = find_free_port()
    print(f"[test B] starting mock server (--view timeline) on :{port} ...")
    proc, log_path, log_file = start_server(port, "mock_server_replay_tl", ["--view", "timeline"])
    try:
        replay(ops, port)
        time.sleep(0.3)
    finally:
        tl_log = stop_server(proc, log_file, log_path)

    print("\n[test B] ----- replayed timeline log -----")
    print(tl_log)
    print("[test B] ----- end of replayed timeline log -----\n")

    def need(marker: str, log: str) -> None:
        nonlocal ok
        if marker not in log:
            print(f"[test B] FAIL: expected {marker!r} in timeline replay log")
            ok = False

    def forbid(marker: str, log: str) -> None:
        nonlocal ok
        if marker in log:
            print(f"[test B] FAIL: {marker!r} must not appear in timeline replay log")
            ok = False

    # lifecycle + attribution + tool events
    need("▶ agent orchestrator", tl_log)
    need("✔ agent orchestrator", tl_log)
    need("▶ agent info_specialist", tl_log)
    need("✔ agent info_specialist", tl_log)
    ts_label = r"\[\d{2}:\d{2}:\d{2}\.\d{3}\]\s+"
    if not re.search(rf"^{ts_label}orchestrator\b.*human\s+{re.escape(HUMAN_ROOT)}", tl_log, re.M):
        print("[test B] FAIL: orchestrator human line missing in timeline log")
        ok = False
    if not re.search(rf"^{ts_label}info_specialist\b.*human\s+{re.escape(HUMAN_SUB)}", tl_log, re.M):
        print("[test B] FAIL: info_specialist human line missing in timeline log")
        ok = False
    if not re.search(rf"^{ts_label}info_specialist\b.*\bai\b.*15C, light rain", tl_log, re.M):
        print("[test B] FAIL: info_specialist ai line missing in timeline log")
        ok = False
    need("⚒ info_specialist_task", tl_log)
    need("⚒ get_weather", tl_log)
    if not re.search(rf"^{ts_label}info_specialist\b.*tool get_weather", tl_log, re.M):
        print("[test B] FAIL: get_weather tool line not attributed to info_specialist")
        ok = False
    need("✖ calculator", tl_log)  # errored run surfaces exactly once
    need(f"·{THREAD}", tl_log)     # thread shown in the label column

    # dedupe: HUMAN_ROOT is carried by 3 runs (root, model, llm) but must be
    # printed exactly once; HUMAN_SUB by 2 runs of info_specialist -> once.
    n_root = tl_log.count(HUMAN_ROOT)
    if n_root != 1:
        print(f"[test B] FAIL: HUMAN_ROOT should appear exactly 1x in timeline "
              f"(deduped), found {n_root}x")
        ok = False
    n_sub_human_lines = len(re.findall(rf"^{ts_label}info_specialist\b.*human\b", tl_log, re.M))
    if n_sub_human_lines != 1:
        print(f"[test B] FAIL: info_specialist should have exactly 1 human line, "
              f"found {n_sub_human_lines}")
        ok = False

    # chain suppression
    for marker in ("START [", "END [", "SESSION CREATED"):
        forbid(marker, tl_log)

    # ---- timeline view over the batched transport -------------------------
    port = find_free_port()
    print(f"[test B] starting mock server (--view timeline, batch ingest) on :{port} ...")
    proc, log_path, log_file = start_server(port, "mock_server_replay_batch", ["--view", "timeline"])
    try:
        replay_as_batch(ops, port)
        time.sleep(0.3)
    finally:
        bt_log = stop_server(proc, log_file, log_path)

    print("\n[test B] ----- replayed batch-ingest timeline log -----")
    print(bt_log)
    print("[test B] ----- end of replayed batch-ingest timeline log -----\n")

    for marker in ("▶ agent orchestrator", "✔ agent orchestrator",
                   "▶ agent info_specialist", "✔ agent info_specialist",
                   "⚒ get_weather", "✖ calculator"):
        need(marker, bt_log)
    if bt_log.count(HUMAN_ROOT) != 1:
        print(f"[test B] FAIL: batched timeline should show HUMAN_ROOT exactly "
              f"1x, found {bt_log.count(HUMAN_ROOT)}x")
        ok = False
    if len(re.findall(rf"^{ts_label}info_specialist\b.*human\b", bt_log, re.M)) != 1:
        print("[test B] FAIL: batched timeline should show exactly 1 "
              "info_specialist human line")
        ok = False
    for marker in ("START [", "END [", "⇈ batch ingest"):
        forbid(marker, bt_log)

    # ---- full view (debug) ------------------------------------------------
    port = find_free_port()
    print(f"[test B] starting mock server (full view) on :{port} ...")
    proc, log_path, log_file = start_server(port, "mock_server_replay_full", [])
    try:
        replay(ops, port)
        time.sleep(0.3)
    finally:
        full_log = stop_server(proc, log_file, log_path)

    print("\n[test B] ----- replayed full log (excerpt) -----")
    print("\n".join(full_log.splitlines()[:40]))
    print("[test B] ----- end of replayed full log -----\n")

    # full mode is the debug view: propagation copies stay visible.
    n_root_full = full_log.count(HUMAN_ROOT)
    if n_root_full < 3:
        print(f"[test B] FAIL: full view should show HUMAN_ROOT >=3x "
              f"(propagation), found {n_root_full}x")
        ok = False
    for marker in ("START [CHAIN] orchestrator", "END [LLM] ChatOpenAI",
                   "END (ERROR)", "agent orchestrator", f"thread {THREAD}"):
        if marker not in full_log:
            print(f"[test B] FAIL: expected {marker!r} in full replay log")
            ok = False
    if "▶ agent" in full_log:
        print("[test B] FAIL: timeline markers must not appear in full view")
        ok = False

    return ok


def main() -> int:
    # Two LLM round trips (~4-8 LLM calls) can take a while; arm the hang
    # diagnoser (no ptrace in this sandbox -- see AGENTS.md).
    faulthandler.dump_traceback_later(240, file=sys.stderr, exit=False)

    ok_a = part_a()
    ok_b = part_b()
    if ok_a and ok_b:
        print("[test] PASS: timeline view verified end-to-end "
              "(agent-labelled, filtered, deduped) and full view unchanged.")
        return 0
    print("[test] FAIL: timeline view E2E incomplete (see diagnostics above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
