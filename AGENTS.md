# AGENTS.md — mock_langsmith_tracer

Project: single-file local mock server for the LangSmith tracing REST API (FastAPI+Uvicorn+Rich),
plus three LangGraph E2E tests (single-agent classic + multi-agent classic + single-agent BATCHED
ingest) against a real remote OpenAI-compatible LLM. STATUS: COMPLETE + verified (all tests exit 0
with "PASS"; multi-agent reproduced 3x, batch test 2x, incl. after the batch/multipart rewrite).

## Layout

```
mock_langsmith_server.py   The mock server (single file). CLI: --port (1984), --max-len (200),
                            --host (127.0.0.1), --width (auto), --raw (full payload dumps),
                            --view {full,timeline} (default full). Endpoints: GET /info,
                            POST /sessions, POST /runs, PATCH /runs/{id}, POST /runs/batch,
                            POST /runs/multipart. ALL run-ingest transports funnel into one
                            shared ingest_run() core (no duplicated logic).
test_e2e_langgraph.py      E2E test: spawns the server as subprocess, runs a LangGraph ReAct agent
                            w/ 2 tools against the real remote LLM, asserts trace markers in server log.
                            Uses classic per-run POST/PATCH (Client(auto_batch_tracing=False)).
test_e2e_multiagent.py     Multi-agent E2E: orchestrator (create_react_agent) + 2 subagents
                            (math_specialist w/ calculator+convert_currency, info_specialist w/
                            get_weather+city_facts), subagents wrapped as @tools. Orchestrator prompt
                            forbids direct answering → immediate parallel delegation (verified: 3
                            tool calls in first turn). Asserts CHAIN/LLM/TOOL markers, both subagent
                            names, >=3 leaf tools, no errored runs, digits in final answer.
test_e2e_batch.py          E2E test using the SDK-DEFAULT batched ingest: plain Client() (no
                            auto_batch_tracing=False) → background queue → POST /runs/batch.
                            Asserts the "⇈ batch ingest: N post · M patch op(s)" summary lines,
                            CHAIN/LLM/TOOL START+END markers, no "BATCHED INGEST IGNORED" stale
                            warning, no errored runs, digits in final answer.
test_e2e_timeline.py       E2E test for --view timeline + agent/thread attribution. Part A (real
                            LLM): langchain.agents.create_agent multi-agent (orchestrator +
                            math_specialist + info_specialist) with MemorySaver checkpointer, TWO
                            invokes on the same thread → asserts ▶/✔ agent lifecycle lines, every
                            human/ai/tool line labelled with agent name + thread id, chain panels
                            suppressed, propagated/history-replayed human text deduped to exactly
                            one human line per turn, no ✖. Part B (deterministic, no LLM): replays
                            a wire-accurate synthetic op tree via classic POST/PATCH, via
                            POST /runs/batch, and against the full view — proves exact dedupe
                            counts (timeline 1x vs full >=3x), subagent attribution, ✖ error
                            lines, and that the full/debug view is unchanged.
requirements.txt           fastapi, uvicorn, rich (server) + langsmith, langchain, langchain-openai,
                            langgraph (test only)
.venv/                     Python 3.14 venv. System pip is externally-managed (PEP 668) — always use
                            a venv here, `pip install --break-system-packages` was NOT needed/used.
                            Confirmed: fastapi/uvicorn/rich/langsmith/langchain/langchain-openai/
                            langgraph all install and run cleanly on 3.14 (unlike /work/crewai-test,
                            where crewai 1.x's tiktoken build breaks on 3.14 — no such issue here).
```

## Server logging design (v2, after multi-agent readability review)

- Default rendering is COMPACT + SEMANTIC, not raw dumps: panels are indented by run nesting depth
  (depth = `dotted_order.count(".")`, fallback: parent_run_id walk via RUN_REGISTRY; indent
  2sp/level, flattening past depth 8). LLM runs render as a chat transcript (`system`/`human`/`ai`/
  `tool` role-colored lines + `↳ ⚒ tool(args)` per tool_call + `tokens in X · out Y · model · finish`
  line). Tool runs show args/result one-liners. Chain runs show only the NEWEST message
  (`(… N earlier not shown)`) + `node X · step Y` from langgraph metadata. Panel subtitle:
  `HH:MM:SS.mmm [. duration] · <last 6 hex of run_id>`. Bodies are padded to min width 48 so
  subtitles never crop. Multi-agent run: ~710 log lines this way vs ~8500 with raw dumps.
- `--raw` restores the old full pretty-printed payload tables. `--width N` forces console width
  (rich defaults to 80 cols when stdout is a file — both E2E tests pass `--width 120`).
- Message extraction handles BOTH LangChain serialization shapes: plain `{"type": "human",
  "content": ...}` (chain/tool payloads) and lc-constructor `{"lc": 1, "type": "constructor",
  "id": [..., "HumanMessage"], "kwargs": {...}}` (LLM inputs/outputs). See `is_message_dict()`,
  `normalize_message()`, `extract_messages()`, `extract_llm_generations()`.
- Tool-run outputs come as `{"output": ToolMessage}` or `{"output": "str"}` — see `_tool_result()`.
- Batched ingest (`POST /runs/batch`, `POST /runs/multipart`) is FULLY IMPLEMENTED and parsed into
  the same panels as classic traffic; each request prints a dim one-line summary first, e.g.
  `⇈ batch ingest: 10 post · 4 patch op(s)` / `⇈ multipart ingest: 1 post · 1 patch op(s) (+1
  feedback ignored)`. Multipart is parsed with a small hand-rolled parser (`_parse_multipart`,
  `_reassemble_multipart_ops`) — `python-multipart` is deliberately NOT a dependency; parts are
  `post.<id>[.field]` / `patch.<id>[.field]` JSON sections. Feedback/attachment parts are counted
  and dropped; zstd-compressed multipart (`Content-Encoding: zstd`) is accepted-with-warning (the
  mock's /info never advertises compression, so it shouldn't occur). The old "BATCHED INGEST
  IGNORED" warning is GONE (that was the pre-implementation state).
- Run durations: computed from payload ISO `start_time`/`end_time` when available (REQUIRED for
  batched traffic — events do not arrive in real time), falling back to arrival-clock deltas
  (classic per-run POST→PATCH). See `_parse_iso_time`, `_run_duration`.
- Panel subtitle shows `run_id`'s LAST 6 hex chars (`_short_id`), NOT the first 8: run ids are
  UUIDv7, so the leading chars are a timestamp and runs created in the same millisecond share the
  same `[:8]` prefix (verified: all of a trace's first-batch runs shared `01a13ffd…`). The
  trailing segment is the random node part and actually disambiguates.
- Catch-all for anything unimplemented (feedback API, datasets, ...) prints a dim
  `(ignored METHOD /path)` line + 200 OK; malformed JSON bodies print a dim warning instead of
  silent `{}`.
- RUN_REGISTRY: entries popped on PATCH / merged-post-end (duration + name/run_type recall), capped
  at 5000 entries; stores parent_run_id + depth + payload start_time for the depth/duration
  fallbacks, plus agent/thread/boundary + tool-run inputs for the timeline view.

## Timeline view (`--view timeline`) + agent/thread attribution (v3)

- `--view {full,timeline}` (default `full`). `full` = the nested panel view described above
  (debugging, whole chain). `timeline` = flat chronological one-line-per-event conversation log:
  `[HH:MM:SS.mmm] <agent> ·<thread> <event>` where event is one of `▶ agent <name>` /
  `✔ agent <name> (dur)` (lifecycle), `human  <text>`, `ai  <text>`, `⚒ tool(args)` (tool calls
  from ai generations), `tool name(args) → result`, `✖ ... ERROR ...` (errored runs). Timestamps
  prefer the payload ISO start/end_time (batched traffic arrives late), else arrival clock.
- **Agent attribution** (`_classify_agent`): primary signal is `lc_agent_name` metadata — set by
  `langchain.agents.create_agent(name=...)` on the compiled graph's config and PROPAGATED to every
  run underneath; a nested subagent re-asserts its own name, so *lc differs from parent's* ==
  subagent boundary. Fallback for graphs without it (e.g. `langgraph.prebuilt.create_react_agent`,
  which compiles with `name=` but sets no lc_agent_name): a chain/agent run at depth 0 is the root
  graph; a chain/agent run whose DIRECT PARENT is a tool run is a subagent graph invoked by that
  tool (unless its name is internal plumbing, see INTERNAL_GRAPH_NAMES). Otherwise inherit the
  parent's agent. Thread = `thread_id` metadata (present when a langgraph checkpointer is used with
  `configurable.thread_id`), inherited from parent when absent. Attribution is computed at POST
  time (parent entry must already be registered — batch posts arrive in creation order, so it is)
  and stored in RUN_REGISTRY; PATCH bodies carry the same metadata for the no-entry fallback.
- **Filtering**: in timeline mode chain/llm/tool/agent panels, the `⇈ batch/multipart ingest`
  summaries, SESSION CREATED panels and catch-all `(ignored ...)` lines are all suppressed — only
  timeline lines (plus startup banner + genuine warnings like malformed JSON) are printed.
- **Dedupe (the "same artifact printed N times as it propagates" fix)**: human messages are scanned
  from EVERY run's inputs; ai generations from llm ends. A per-(agent, thread) seen-set
  (`TIMELINE_SEEN`, key = role + squashed content + sorted (tool_name, tool_id) pairs, capped
  20000) makes each message print exactly once per agent+thread no matter how many runs carried it
  — covers chain→llm propagation copies AND checkpointer history replays on later turns of the
  same thread. NOT keyed by trace_id (a new invoke on the same thread must not re-print old
  history). The same text under a DIFFERENT agent (e.g. delegation task text) legitimately prints
  once per agent. `full` view deliberately does NOT dedupe (debugging).
- Full view additionally shows `agent <name> · thread <id>` in every panel's meta line
  (`_meta_line`), so attribution is visible in both views.
- Line grammar is greppable: tests target `▶ agent`, `✔ agent`, `human`, `ai`, `⚒`, `tool`, `✖`
  and per-line label regexes `^\[\d{2}:\d{2}:\d{2}\.\d{3}\]\s+<agent>\b.*` (label column is padded,
  agent names get a deterministic color via sum-of-ord hash, not python's randomized hash()).

## LangSmith SDK wire protocol — key facts (langsmith==0.14.8)

- By default `Client(auto_batch_tracing=True)` (the default used by `LangChainTracer`/env-var-based
  tracing) buffers runs and ships them via **multipart** (`POST /runs/multipart`) or **batch**
  (`POST /runs/batch`) endpoints — NOT the classic per-run `POST /runs` + `PATCH /runs/{id}`.
- To get the simple classic REST shape (what this task required), construct the client explicitly with
  `Client(auto_batch_tracing=False)`. This makes `tracing_queue=None` → every run start/end is sent
  **synchronously**, individually, as `POST /runs` then `PATCH /runs/{id}`. Pass this client into
  `langchain_core.tracers.context.tracing_v2_enabled(project_name=..., client=client)` (or
  `LangChainTracer(client=client)`), do NOT rely solely on `LANGCHAIN_TRACING_V2=true` env var picking
  it up, since the env-var path builds its own default client (batched).
- `GET /info` → `LangSmithInfo` (all fields optional, `{}` is a valid response). Can advertise
  `batch_ingest_config.use_multipart_endpoint: false` to further bias behavior, but it's the
  constructor's `auto_batch_tracing` flag that actually matters for whether POST/PATCH are
  synchronous+per-run vs batched.
- `POST /sessions` response needs `TracerSession` schema fields: `id`, `tenant_id` required (UUID),
  `reference_dataset_id` required-but-nullable (must be present, even if `null`), `start_time`
  defaults OK. In practice `LangChainTracer`/`RunTree.post()` never calls `/sessions` on its own —
  the project is implicitly created server-side from `session_name` inside the `POST /runs` body.
  Endpoint only gets hit if the test/user explicitly calls `client.create_project()`.
- `POST /runs` body (from `Client._run_transform`) contains: `id`, `name`, `run_type`, `inputs`,
  `outputs`, `error`, `session_name`, `start_time`, `end_time`, `extra` (has `.metadata`), `tags`,
  `trace_id`, `dotted_order`, `parent_run_id`. `PATCH /runs/{id}` body may omit `name`/`run_type` —
  cache them from the POST side (by run `id`) if you need them at PATCH time.
- **Batched ingest wire format** (`POST /runs/batch`): body is `{"post": [<run dict>, ...],
  "patch": [<run dict>, ...]}` — each entry is exactly the same run dict the classic endpoints
  get (patch entries also carry `id` in the body). Verified by capturing real traffic
  (recording HTTP server + tracer, see /tmp/opencode/capture_e2e_batch.py pattern):
  - **`end_time` present in a POST entry ⟺ that entry is a COMPLETE run** — either its patch was
    merged into the post (`combine_serialized_queue_operations` merges same-queue-batch patches
    into their posts; the background queue drains every 250ms) or the tracer deferred the post
    until run end (langchain_core `defers_inputs` runs, e.g. langgraph's `Prompt`/`should_continue`
    nodes, are ONLY posted at end via `_on_chain_end`→`_persist_run_single`). Log START+END for
    such posts.
  - **CRITICAL: fresh posts DO carry `"outputs": {}`** (RunTree's default, `_get_dicts_safe`
    copies it even at start) — `outputs is not None` must NEVER be used as the "run ended" signal,
    it produces phantom END panels. `end_time is not None` is the only reliable signal.
  - Patch-only fields come with `end_time` always set (client `update_run` sets it); `start_time`
    is NOT in patch bodies → registry must store the post's `start_time` for ISO duration math.
  - Fast runs (tools, graph nodes) usually merge post+patch into one complete entry per run; slow
    runs (LLM calls) arrive as post in one batch request + patch in a later one.
- **Multipart wire format** (`POST /runs/multipart`): form parts named `post.<run_id>` /
  `patch.<run_id>` (main run dict minus big fields) + `post.<run_id>.inputs|.outputs|.events|
  .extra|.error|.serialized` (each a separate JSON part), `attachment.<run_id>.<name>` (binary),
  `feedback.<id>` (JSON). Only used when `/info` advertises
  `batch_ingest_config.use_multipart_endpoint: true` (the SDK's client-side default if absent!)
  or on explicit `Client.multipart_ingest()`; a 404 makes the SDK fall back to `/runs/batch`
  and disable multipart. This mock advertises `false`, so batched clients use `/runs/batch`.
  SDK chooses batch-vs-multipart in `tracing_control_thread_func`
  (`langsmith/_internal/_background_thread.py` ~L552).
- Useful source map (read before touching this again): `langsmith/client.py` —
  `create_run`/`_create_run`/`_create_run_non_batch` (~L2572-2960), `update_run`/`_update_run`
  (~L3828-4135), `_run_transform` (~L2449), `_batch_ingest_run_ops` (~L3091),
  `_multipart_ingest_ops` (~L3400), `create_project` (~L5383); `langsmith/run_trees.py` —
  `post`/`patch`/`_get_dicts_safe` (~L814-1110); `langsmith/_internal/_operations.py` —
  `serialize_run_dict`/`combine_serialized_queue_operations`; `langchain_core/tracers/langchain.py`
  — `_persist_run_single`/`_update_run_single`/`_on_chain_end` (defers_inputs post-at-end path).
  Confirmed experimentally by grepping the installed venv package + capturing real traffic rather
  than guessing from docs.

## CRITICAL BUG CLASS: subprocess.PIPE deadlock (cost ~20 min to diagnose — avoid repeating)

Symptom: test script that spawns the mock server via `subprocess.Popen(..., stdout=subprocess.PIPE)`
and only reads that pipe **after** the whole test finishes will **silently deadlock forever** once the
server's stdout output exceeds the OS pipe buffer (~64KB on Linux). Rich's verbose panel dumps
(full LangGraph message histories, etc.) blow past that fast. The server blocks on `write()`
mid-request, which blocks the HTTP response, which blocks every synchronous SDK call made against it
— looks exactly like a hung LLM call, NOT like a pipe problem, because the last visible print is
"invoking agent...". Diagnosis took: `ps`/`ss -tnp` (showed established connections, so not a network
timeout), then `faulthandler.dump_traceback_later(N, file=sys.stderr, exit=False)` inside the hung
script (no root/ptrace available in this sandbox for `py-spy`/`strace -p`, both require elevated
perms not available here) — the dumped traceback pinpointed the exact blocking frame
(`http.client._read_status` inside `langsmith.client.request_with_retries` → `_create_run_non_batch`).
**FIX**: redirect subprocess stdout to a real file (`open(path, "w")`), not `subprocess.PIPE`, when you
don't actively drain the pipe in real time. This is the single most important lesson from this session
— apply it to ANY test harness here that spawns a chatty subprocess and only checks output at the end.

## Debugging tools available/unavailable in this sandbox

- No passwordless `sudo` (`sudo -n true` fails, needs real TTY). No `ptrace` permission for own user
  (`ptrace_scope=1`): both `strace -p <pid>` and `py-spy dump --pid <pid>` fail with permission errors.
- **`faulthandler.dump_traceback_later(seconds, file=sys.stderr, exit=False)`** (stdlib, add at top of
  `main()`) is the go-to hang-diagnosis tool here since ptrace-based tools don't work — dumps every
  thread's Python stack after a timeout, no special perms needed.
- `ss -tnp | grep <pid>` to check live socket state (ESTAB vs CLOSE-WAIT) is useful to rule out
  "still waiting on remote" vs "already got response, stuck in local code".

## systemd-run-background skill — gotchas observed this session

- Essential for long-running LLM/E2E calls since the `bash` tool's own command timeout (default/max
  ~120s) is too short for a full agent.invoke() + trace flush cycle (~8-20s is fine, but margin is
  thin once you add process spawn/teardown).
- Observed repeatedly: trivial, fast commands (`pkill -9 -f ...`, `systemctl --user kill ...`, even a
  plain `ps aux | grep ...` chained after them) sometimes made the **bash tool itself** hang for the
  full 120s timeout with no apparent cause, even though the underlying OS action (process already
  dead) completed instantly. Root cause unconfirmed (not a real hang — likely tool/shell-wrapper
  flakiness unrelated to the command semantics). **Workaround**: when a bash call reports a timeout,
  don't assume the system is in a bad state — immediately re-issue a trivial follow-up command (e.g.
  `echo alive` or `ps aux | grep ...`) to confirm the shell and processes are actually fine. In every
  occurrence this session, the target process/unit was already gone/dead despite the tool-level
  timeout.
- `systemd-run --user -u <name> -d` then `sleep N; journalctl --user -u <name> -n 300 --no-pager`
  in follow-up bash calls is the reliable polling pattern for a slow background job.
  `systemctl --user show <name> --property=ActiveState,SubState,ExecMainStatus` for a quick
  pass/fail/still-running check without full logs.
  Must `systemctl --user reset-failed <name>` before reusing a unit name (even if previously
  stopped cleanly, reuse sometimes still complains).
- journalctl only shows what the unit's own stdout emits — a subprocess spawned BY the unit's
  process (e.g. our mock server, launched via `subprocess.Popen` from inside the python test) does
  NOT appear in `journalctl -u <unit>` unless its fd is inherited; we captured it separately via
  file redirect + the test script printing it out itself at the end.

## Verified external facts (LLM endpoint, reusable across sessions)

- `http://open-webai.kt53.net/openai` (OpenAI-compatible, reachable from this sandbox).
  `/models` lists `qwen3-8-27b` (works, tool-calling confirmed via raw curl AND via
  langchain_openai ChatOpenAI + langgraph create_react_agent), `qwen-3-8-flash-next`, plus many
  others (claude-opus-5, gte-qwen2-7b, bge-reranker-v2-m3, ...). API key used:
  `sk-024610f905ea4b4d8eb4cb0b34f6e811` (same key seen reused in /work/crewai-test/AGENTS.md for a
  DIFFERENT model `qwen-3-8-flash-next` with an `openai/` prefix quirk for CrewAI — that prefix
  quirk is CrewAI/LiteLLM-specific and does NOT apply here; `langchain_openai.ChatOpenAI` takes the
  bare model name `qwen3-8-27b`, no prefix).
- `qwen3-8-27b` responses include a `reasoning_content` field and often an empty/whitespace-only
  `content` on tool-call turns (e.g. `"content": "\n\n"`) — normal, not a bug; final answer turn has
  real content.
  Full agent.invoke() round trip (1 reasoning turn + 2 parallel tool calls + final answer) takes
  ~6-8s wall clock in this sandbox.
- `openaipublic.blob.core.windows.net` (tiktoken encoding downloads) is reachable — ruled out as a
  hang cause early, but worth knowing it's one of the first things to suspect for a "ChatOpenAI hangs
  on first call in a sandbox" symptom (silent token-counting download) if it ever recurs elsewhere.

## Verification

- Run `python3 test_e2e_langgraph.py` (via systemd-run — see above, not directly in foreground bash,
  it takes ~8-20s including process spawn/teardown which can wobble close to tool timeouts).
  PASS criterion: stdout contains `[test] PASS: mock server observed LLM and TOOL run traces
  end-to-end.` and process exit code 0. Confirmed reproducible across 3 separate runs.
- Run `python3 test_e2e_multiagent.py` the same way (allow ~60-90s; ~8 LLM calls incl. 2 parallel
  subagent runs). PASS criterion: stdout contains `[test] PASS: mock server observed the full
  multi-agent run tree end-to-end.` and exit code 0. Confirmed across 3 separate runs (incl. once
  after the v2 logging rewrite). Run count note: ~8-10 LLM round trips ≈ 25-35s wall clock.
- Run `python3 test_e2e_batch.py` the same way (~15-20s). PASS criterion: stdout contains
  `[test] PASS: mock server observed the batched (runs/batch) run tree end-to-end.` and exit code
  0. Confirmed across 2 separate runs (incl. after the phantom-END fix). This is the test for the
  SDK-DEFAULT mode (plain `Client()` → background queue → `POST /runs/batch`).
- Run `python3 test_e2e_timeline.py` the same way (~35-60s; Part A does two LLM turns on one
  thread, ~6-9 LLM round trips; Part B is a fast deterministic replay, no LLM). PASS criterion:
  stdout contains `[test] PASS: timeline view verified end-to-end (agent-labelled, filtered,
  deduped) and full view unchanged.` and exit code 0. Confirmed across 2 separate full runs
  (plus repeated Part B-only runs during development). Part B alone can be run with
  `python3 -c "import test_e2e_timeline as t, sys; sys.exit(0 if t.part_b() else 1)"`.
- Manual spot-check recipe for the server alone (no LLM/E2E needed): a ready-made no-LLM smoke
  script exercising classic POST/PATCH + `/runs/batch` (incl. a merged post with `end_time` +
  a fresh post with `outputs: {}` that must NOT phantom-END) + `/runs/multipart` (incl. a
  feedback part) lives at /tmp/opencode/smoke_batch.py — regenerate if tmp is wiped. It starts
  its own server, asserts markers + ISO-duration rendering, exits non-zero on failure. Plain
  curl also works: background-launch `mock_langsmith_server.py --port <p>`, then `curl` the
  endpoints with JSON bodies — no network/LLM access needed at all.

## Plan / TODOs (not started, optional follow-ups)

1. No automated regression test for the truncation function itself (`truncate()` in the server) —
   only exercised informally via manual curl with long strings. Could add a small unit test.
2. `/runs/multipart` is implemented + smoke-tested with hand-built bodies but has NO LLM E2E
   (the SDK only picks it when /info advertises `use_multipart_endpoint: true`, which this mock
   deliberately doesn't). Flip that flag in `/info` + clone test_e2e_batch.py if real-traffic
   coverage for it is ever wanted.
