#!/usr/bin/env python3
"""
mock_langsmith_server.py
========================

A minimal, single-file, *local* mock server that speaks just enough of the
LangSmith (and compatible LangFuse-style) tracing protocol for the
``langsmith`` Python SDK -- and anything built on top of it (``langchain``,
``langgraph``, etc.) -- to trace against it instead of the real
smith.langchain.com backend.

It implements every endpoint the tracing SDK needs, for BOTH wire
protocols -- the classic per-run REST shape (``Client(auto_batch_tracing=
False)``) and the batched ingest shape (the SDK default, buffers runs and
ships them in bulk):

    GET   /info                -> server capability banner
    POST  /sessions            -> "create project" (dummy id returned)
    POST  /runs                -> a run started (classic per-run protocol)
    PATCH /runs/{run_id}       -> a run finished (classic per-run protocol)
    POST  /runs/batch          -> batched ingest: {"post": [...], "patch": [...]}
    POST  /runs/multipart      -> multipart/form-data variant of the same

All four run-ingesting transports funnel into a single shared ``ingest_run()``
core, so run parsing / registry / rendering logic exists exactly once.

Batched-ingest notes: the SDK may merge a run's patch into its post when
both are queued in the same client-side batch (such a post already carries
``end_time`` / ``outputs`` / ``error`` -- the mock then logs START *and*
END from the single entry). Run durations are computed from the payload's
ISO ``start_time`` / ``end_time`` (batched events do not arrive in real
time), falling back to arrival-time deltas for classic per-run traffic.

All authentication (``x-api-key``, ``Authorization``, etc.) is completely
ignored -- there is no real auth, this is only meant for local development.
Anything else the SDK might probe (feedback, datasets, ...) hits a
catch-all returning 200 OK.

Console output (rich) supports two views (``--view``):

``full`` (default) -- every run start/end is printed as a compact,
color-coded panel that is *indented by its nesting depth* (derived from
the run's ``dotted_order``), so parent/child structure is visible at a
glance. Designed for debugging the whole chain:

  * LLM runs are rendered as a chat transcript (``system`` / ``human`` /
    ``ai`` / ``tool`` lines, color-coded per role, with tool calls listed
    inline) instead of raw thousand-line JSON dumps.
  * Tool runs show their arguments and result on one line each.
  * Chain runs show the newest message of their input/output plus the
    langgraph node/step, instead of re-dumping the whole history.
  * Every panel's meta line shows the owning agent (``lc_agent_name``)
    and langgraph thread (``thread_id``) when the client provides them.
  * Every panel carries a timestamp, duration (on end) and short run id.
  * All strings are recursively truncated to ``--max-len`` characters.
  * Pass ``--raw`` to get the original behaviour back (full pretty-printed
    payloads, no semantic rendering).

``timeline`` -- a high-level, human-readable *conversation timeline*,
boiled down to what a human cares about when watching agents work:
agent lifecycle (start/end), all human/user and LLM text messages, tool
calls and tool results, and errors -- each line prefixed with the owning
agent's name (``lc_agent_name``, falling back to the enclosing named
graph) and its langgraph ``thread_id`` when available. Concurrently
running agents are fine: every line is labelled, so interleaved output
stays unambiguous. Internal chain plumbing (graph nodes, prompt/sequence
runnables, ...) is NOT printed in this view, and *message artifacts are
deduplicated*: the same message propagated through several runs of one
agent+thread (chain in -> llm in -> ...) is emitted exactly once.

Usage
-----

    python3 mock_langsmith_server.py --port 1984 --max-len 200 [--width 120]
                                     [--raw] [--view {full,timeline}]

Then point the LangSmith/LangChain SDK at it:

    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_ENDPOINT=http://127.0.0.1:1984
    export LANGCHAIN_API_KEY=anything-goes
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text

# --------------------------------------------------------------------------- #
# Globals configured from CLI args (set in main())
# --------------------------------------------------------------------------- #

MAX_LEN = 200           # overwritten by --max-len
RAW_PAYLOADS = False    # overwritten by --raw
VIEW = "full"           # overwritten by --view ("full" panels | "timeline")
console = Console()

# run_id -> {"name", "run_type", "start" (arrival clock), "start_time" (ISO
# epoch from the payload), "parent_run_id", "depth", "agent", "thread",
# "boundary", "tool_args"}
# Lets us recall a run's identity/duration when the end event arrives (PATCH
# bodies don't always repeat name/run_type), compute accurate durations for
# batched transports, fall back to parent-chain walking for depth, and keep
# the agent/thread context + tool arguments needed by the timeline view.
RUN_REGISTRY: dict[str, dict[str, Any]] = {}
RUN_REGISTRY_MAX = 5000  # safety valve against unbounded growth

# Chain/graph run names that are internal plumbing rather than a named
# (sub-)agent graph. Used only for the "chain invoked directly by a tool"
# boundary heuristic when no lc_agent_name metadata is available.
INTERNAL_GRAPH_NAMES = {
    "agent", "tools", "model", "call_model", "should_continue",
    "LangGraph", "Prompt", "RunnableSequence", "RunnableLambda",
    "RunnableParallel", "RunnableAssign", "RunnableMap", "bind", "bind_tools",
}

# Timeline view state: per (trace, agent, thread) dedupe of messages that
# propagate unchanged through several runs of the same agent (chain in ->
# llm in -> ...), so each artifact is emitted exactly once.
TIMELINE_SEEN: set[tuple[Any, ...]] = set()
TIMELINE_SEEN_MAX = 20000
TL_LABEL_WIDTH = 18  # grows to the longest agent label actually seen

# Color / icon scheme per run_type. Anything unlisted falls back to "white".
RUN_TYPE_STYLES: dict[str, dict[str, str]] = {
    "chain":     {"color": "magenta", "icon": "⛓ "},
    "llm":       {"color": "blue",    "icon": "🧠"},
    "tool":      {"color": "green",   "icon": "🔧"},
    "agent":     {"color": "cyan",    "icon": "🤖"},
    "retriever": {"color": "yellow",  "icon": "📚"},
    "prompt":    {"color": "bright_blue", "icon": "📝"},
    "parser":    {"color": "bright_green", "icon": "🧩"},
    "embedding": {"color": "bright_magenta", "icon": "🔗"},
}
DEFAULT_STYLE = {"color": "white", "icon": "•"}

# Color scheme per chat message role (for LLM transcript rendering).
ROLE_STYLES: dict[str, str] = {
    "system": "dim",
    "developer": "dim",
    "human": "cyan",
    "ai": "magenta",
    "tool": "green",
    "function": "green",
}
ROLE_ALIASES = {"user": "human", "assistant": "ai"}
KNOWN_ROLES = {"system", "developer", "human", "ai", "tool", "function",
               "user", "assistant", "moderation", "chat"}


def style_for(run_type: str | None) -> dict[str, str]:
    return RUN_TYPE_STYLES.get((run_type or "").lower(), DEFAULT_STYLE)


# --------------------------------------------------------------------------- #
# Recursive truncation helpers
# --------------------------------------------------------------------------- #

def truncate(value: Any, max_len: int, _depth: int = 0) -> Any:
    """Recursively walk ``value`` (dict / list / str / anything) and ensure
    that no string anywhere in the structure is longer than ``max_len``
    characters. Also caps recursion depth and collection length so that
    deeply nested or very wide structures still produce bounded output.
    """
    if _depth > 12:
        return "<...max depth...>"

    if isinstance(value, str):
        if len(value) > max_len:
            return value[:max_len] + f"... <+{len(value) - max_len} chars>"
        return value

    if isinstance(value, dict):
        out = {}
        items = list(value.items())
        shown, rest = items[:50], items[50:]
        for k, v in shown:
            out[k] = truncate(v, max_len, _depth + 1)
        if rest:
            out[f"...+{len(rest)} more keys"] = "..."
        return out

    if isinstance(value, (list, tuple)):
        shown, rest = value[:50], value[50:]
        out = [truncate(v, max_len, _depth + 1) for v in shown]
        if rest:
            out.append(f"...+{len(rest)} more items")
        return out

    # numbers, bools, None, and anything else rich can pretty-print directly
    return value


def squash(text: str) -> str:
    """Collapse all whitespace runs into single spaces (prompts and chat
    contents are multi-line; for a one-line-per-message log view we want
    them on a single line)."""
    return re.sub(r"\s+", " ", text or "").strip()


def compact_line(value: Any, cap: int | None = None) -> str:
    """Render a small payload (tool args, fallback blobs) as a single-line
    JSON string, truncated per-string by ``truncate`` and capped in total."""
    if value is None:
        return "-"
    try:
        s = json.dumps(truncate(value, MAX_LEN), default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(truncate(value, MAX_LEN))
    s = squash(s)
    cap = cap or max(400, MAX_LEN * 3)
    if len(s) > cap:
        s = s[:cap] + f" …(+{len(s) - cap} chars)"
    return s


# --------------------------------------------------------------------------- #
# Chat message extraction (LangChain serializes messages in two shapes:
# plain dicts {"type": "human", "content": ...} and lc-constructor dicts
# {"lc": 1, "type": "constructor", "id": [..., "HumanMessage"], "kwargs": {...}}.
# Both appear in LLM run inputs/outputs; the plain shape appears in chain
# and tool run payloads.)
# --------------------------------------------------------------------------- #

def is_message_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("lc") == 1 and obj.get("type") == "constructor" \
            and isinstance(obj.get("id"), list) and obj["id"] \
            and str(obj["id"][-1]).endswith("Message"):
        return True
    return (obj.get("type") in KNOWN_ROLES) or (obj.get("role") in KNOWN_ROLES)


def normalize_message(obj: dict) -> dict:
    kwargs = {}
    if obj.get("lc") == 1 and obj.get("type") == "constructor":
        kwargs = obj.get("kwargs") or {}
        role = kwargs.get("type")
        if not role and isinstance(obj.get("id"), list) and obj["id"]:
            role = str(obj["id"][-1]).removesuffix("Message").lower()
    else:
        role = obj.get("type") or obj.get("role")
    role = ROLE_ALIASES.get(role, role) or "?"

    content = kwargs.get("content", obj.get("content"))
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            elif isinstance(block, str):
                parts.append(block)
        content = " ".join(p for p in parts if p)
    if not isinstance(content, str):
        content = "" if content is None else str(content)

    tool_calls = kwargs.get("tool_calls", obj.get("tool_calls")) or []

    # usage / model / finish reason live in different spots per shape
    usage = kwargs.get("usage_metadata") or obj.get("usage_metadata") or None
    resp_meta = kwargs.get("response_metadata") or obj.get("response_metadata") or {}
    return {
        "role": str(role),
        "content": content,
        "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
        "usage": usage if isinstance(usage, dict) else None,
        "model": resp_meta.get("model_name") if isinstance(resp_meta, dict) else None,
        "finish": resp_meta.get("finish_reason") if isinstance(resp_meta, dict) else None,
    }


def extract_messages(obj: Any, _depth: int = 0) -> list[dict]:
    """Recursively find every chat-message-like dict inside ``obj``
    (LLM inputs nest them as {'messages': [[...]]}, chain payloads as
    {'messages': [...]}, etc.)."""
    found: list[dict] = []
    if _depth > 8:
        return found
    if is_message_dict(obj):
        return [normalize_message(obj)]
    if isinstance(obj, dict):
        for v in obj.values():
            found.extend(extract_messages(v, _depth + 1))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(extract_messages(v, _depth + 1))
    return found


def extract_llm_generations(outputs: Any) -> list[dict]:
    """Pull the generated AIMessages out of an LLM run's outputs
    (``{'generations': [[{'message': {...}, ...}], ...], ...}``)."""
    gens: list[dict] = []
    if isinstance(outputs, dict) and isinstance(outputs.get("generations"), list):
        for batch in outputs["generations"]:
            if not isinstance(batch, list):
                continue
            for gen in batch:
                if not isinstance(gen, dict):
                    continue
                msg = gen.get("message")
                if is_message_dict(msg):
                    gens.append(normalize_message(msg))
                elif squash(gen.get("text") or ""):
                    gens.append(normalize_message({"type": "ai", "content": gen.get("text")}))
    return gens


# --------------------------------------------------------------------------- #
# Pretty console logging
# --------------------------------------------------------------------------- #

def _timestamp() -> str:
    dt = datetime.now()
    return f"{dt:%H:%M:%S}.{dt.microsecond // 1000:03d}"


def _short_id(run_id: str) -> str:
    """Run ids are UUIDv7: the leading hex chars are a timestamp, so runs
    created in the same millisecond share the same prefix. Show the trailing
    random segment instead -- it actually disambiguates runs."""
    return run_id.replace("-", "")[-6:]


def _indent_for(depth: int) -> int:
    # 2 columns per nesting level, gently flattening for very deep runs so
    # that panels keep enough horizontal space for their content.
    return 2 * min(depth, 8) + max(0, min(depth, 16) - 8)


MIN_BODY_WIDTH = 48  # keep panels wide enough that title+subtitle never crop


def _print_panel(depth: int, body_lines: list[str], *, title: str,
                 border_color: str, subtitle: str) -> None:
    if body_lines:
        # Pad the body to a minimum visible width so that narrow panels
        # don't squeeze/crop the title and subtitle.
        texts = [Text.from_markup(line) for line in body_lines]
        pad_to = max(MIN_BODY_WIDTH, max(t.cell_len for t in texts))
        for t in texts:
            t.pad_right(pad_to - t.cell_len)
        body: Any = Group(*texts)
    else:
        body = "[dim](no payload)[/dim]"
    console.print(
        Padding(
            Panel(
                body,
                title=title,
                subtitle=subtitle,
                border_style=border_color,
                expand=False,
            ),
            (0, 0, 0, _indent_for(depth)),
        )
    )


def _panel_title(icon: str, mark: str, run_type: str, name: str, color: str) -> str:
    rt = (run_type or "?").upper()
    return (f"[bold {color}]{icon} {mark} [{rt}][/bold {color}] "
            f"[bold white]{name}[/bold white]")


def _render_message(msg: dict, lead: str = "") -> list[str]:
    """One line per message (role-colored), one line per tool call."""
    role = msg["role"]
    style = ROLE_STYLES.get(role, "white")
    content = squash(msg["content"])
    if not content:
        content = (f"(no text · {len(msg['tool_calls'])} tool call(s))" if msg["tool_calls"]
                   else "(empty)")
    lines = [f"[{style}]{role:<8}[/{style}] ⋅ {truncate(content, MAX_LEN)}"]
    for tc in msg["tool_calls"][:6]:
        if not isinstance(tc, dict):
            continue
        lines.append(
            f"[dim]{lead}         ↳ ⚒[/dim] [green]{tc.get('name', '?')}[/green]"
            f"({compact_line(tc.get('args'), cap=320)})"
        )
    if len(msg["tool_calls"]) > 6:
        lines.append(f"[dim]{lead}         ↳ … +{len(msg['tool_calls']) - 6} more tool calls[/dim]")
    return lines


def _last_message_line(label: str, payload: Any) -> list[str]:
    """For chain-ish payloads: show only the *newest* message instead of
    re-dumping the whole history at every nesting level."""
    msgs = extract_messages(payload)
    if not msgs:
        if payload is None:
            return []
        return [f"[dim]{label}[/dim] {compact_line(payload)}"]
    lines = [f"[dim]{label}[/dim] " + _render_message(msgs[-1])[0]]
    # tool-call detail of the newest message, if any
    detail = _render_message(msgs[-1])[1:]
    lines.extend(f"         {ln}" for ln in detail)
    if len(msgs) > 1:
        lines.append(f"[dim]          (… {len(msgs) - 1} earlier message(s) not shown)[/dim]")
    return lines


def _meta_line(extra: dict | None, agent: str | None = None,
               thread: str | None = None) -> list[str]:
    bits = []
    if agent:
        bits.append(f"agent {agent}")
    if thread:
        bits.append(f"thread {thread}")
    if not isinstance(extra, dict):
        return [f"[dim]{' · '.join(bits)}[/dim]"] if bits else []
    meta = extra.get("metadata")
    if not isinstance(meta, dict):
        return [f"[dim]{' · '.join(bits)}[/dim]"] if bits else []
    if meta.get("langgraph_node"):
        bits.append(f"node {meta['langgraph_node']}")
    if meta.get("langgraph_step") is not None:
        bits.append(f"step {meta['langgraph_step']}")
    return [f"[dim]{' · '.join(bits)}[/dim]"] if bits else []


def log_run_start(run_id: str, name: str, run_type: str, inputs: Any,
                  extra: dict | None, depth: int,
                  agent: str | None = None, thread: str | None = None) -> None:
    st = style_for(run_type)
    color, icon = st["color"], st["icon"]
    rt = (run_type or "").lower()

    if RAW_PAYLOADS:
        table = Table.grid(padding=(0, 1))
        table.add_column()
        table.add_column()
        table.add_row("[bold]id[/bold]", f"[dim]{run_id}[/dim]")
        table.add_row("[bold]inputs[/bold]", _fmt_payload_block(inputs))
        if extra and extra.get("metadata"):
            table.add_row("[bold]metadata[/bold]", _fmt_payload_block(extra.get("metadata")))
        console.print(
            Padding(
                Panel(table,
                      title=_panel_title(icon, "▶ START", run_type, name, color),
                      subtitle=f"[dim]{_timestamp()} · {_short_id(run_id)}[/dim]",
                      border_style=color, expand=False),
                (0, 0, 0, _indent_for(depth)),
            )
        )
        return

    lines: list[str] = []
    if rt == "llm":
        msgs = extract_messages(inputs)
        if msgs:
            lines.append(f"[dim]{len(msgs)} input message(s)[/dim]")
            if len(msgs) > 7:
                for m in msgs[:2]:
                    lines.extend(_render_message(m))
                lines.append(f"[dim]⋯ {len(msgs) - 7} message(s) omitted ⋯[/dim]")
                for m in msgs[-5:]:
                    lines.extend(_render_message(m))
            else:
                for m in msgs:
                    lines.extend(_render_message(m))
        else:
            lines.append(f"[dim]in[/dim] {compact_line(inputs)}")
    elif rt == "tool":
        lines.append(f"[dim]args[/dim] {compact_line(inputs)}")
    else:  # chain / agent / retriever / ...
        lines.extend(_last_message_line("in   ", inputs))
    lines.extend(_meta_line(extra, agent, thread))

    _print_panel(
        depth, lines,
        title=_panel_title(icon, "▶ START", run_type, name, color),
        border_color=color,
        subtitle=f"[dim]{_timestamp()} · {_short_id(run_id)}[/dim]",
    )


def log_run_end(run_id: str, name: str, run_type: str, outputs: Any, error: Any,
                duration_s: float | None, depth: int,
                agent: str | None = None, thread: str | None = None) -> None:
    st = style_for(run_type)
    color, icon = st["color"], st["icon"]
    rt = (run_type or "").lower()
    failed = error is not None

    if RAW_PAYLOADS:
        table = Table.grid(padding=(0, 1))
        table.add_column()
        table.add_column()
        table.add_row("[bold]id[/bold]", f"[dim]{run_id}[/dim]")
        if duration_s is not None:
            table.add_row("[bold]duration[/bold]", f"{duration_s:.3f}s")
        if failed:
            table.add_row("[bold red]error[/bold red]", _fmt_payload_block(error))
        else:
            table.add_row("[bold]outputs[/bold]", _fmt_payload_block(outputs))
        header_color = "red" if failed else color
        mark = "✖ END (ERROR)" if failed else "✔ END"
        dur = f" · {duration_s:.2f}s" if duration_s is not None else ""
        console.print(
            Padding(
                Panel(table,
                      title=_panel_title(icon, mark, run_type, name, header_color),
                      subtitle=f"[dim]{_timestamp()}{dur} · {_short_id(run_id)}[/dim]",
                      border_style=header_color, expand=False),
                (0, 0, 0, _indent_for(depth)),
            )
        )
        return

    lines: list[str] = []
    if failed:
        lines.append(f"[bold red]error[/bold red] {truncate(squash(str(error)), MAX_LEN)}")
    elif rt == "llm":
        gens = extract_llm_generations(outputs)
        if gens:
            for g in gens[:3]:
                detail = _render_message(g)
                lines.append(f"[dim]out[/dim] {detail[0]}")
                lines.extend(f"         {ln}" for ln in detail[1:])
            if len(gens) > 3:
                lines.append(f"[dim]         … +{len(gens) - 3} more generation(s)[/dim]")
            usage = next((g["usage"] for g in gens if g.get("usage")), None)
            model = next((g["model"] for g in gens if g.get("model")), None)
            finish = next((g["finish"] for g in gens if g.get("finish")), None)
            bits = []
            if isinstance(usage, dict):
                bits.append(f"tokens in {usage.get('input_tokens', '?')} · out {usage.get('output_tokens', '?')}")
            if model:
                bits.append(str(model))
            if finish:
                bits.append(f"finish {finish}")
            if bits:
                lines.append(f"[dim]{' · '.join(bits)}[/dim]")
        else:
            lines.append(f"[dim]out[/dim] {compact_line(outputs)}")
    elif rt == "tool":
        lines.append(f"[dim]out[/dim] {_tool_result(outputs)}")
    else:  # chain / agent / retriever / ...
        lines.extend(_last_message_line("out  ", outputs))
    lines.extend(_meta_line(None, agent, thread))

    header_color = "red" if failed else color
    mark = "✖ END (ERROR)" if failed else "✔ END"
    dur = f" · {duration_s:.2f}s" if duration_s is not None else ""
    _print_panel(
        depth, lines,
        title=_panel_title(icon, mark, run_type, name, header_color),
        border_color=header_color,
        subtitle=f"[dim]{_timestamp()}{dur} · {_short_id(run_id)}[/dim]",
    )


def _tool_result(outputs: Any) -> str:
    """Tool run outputs come in several shapes: {'output': ToolMessage},
    {'output': 'plain string'}, a bare string, or something else."""
    if outputs is None:
        return "-"
    if isinstance(outputs, dict) and "output" in outputs and not isinstance(outputs.get("messages"), list):
        val = outputs["output"]
        if is_message_dict(val):
            return truncate(squash(normalize_message(val)["content"]), MAX_LEN) or "(empty)"
        if isinstance(val, str):
            return truncate(squash(val), MAX_LEN)
        return compact_line(val)
    if isinstance(outputs, str):
        return truncate(squash(outputs), MAX_LEN)
    msgs = extract_messages(outputs)
    if msgs:
        return truncate(squash(msgs[-1]["content"]), MAX_LEN) or "(empty)"
    return compact_line(outputs)


def _fmt_payload_block(payload: Any) -> Pretty | str:
    if payload is None:
        return "-"
    return Pretty(truncate(payload, MAX_LEN), max_string=MAX_LEN)


# --------------------------------------------------------------------------- #
# Timeline view (--view timeline): a flat, chronological, one-line-per-event
# log of the *conversation* -- agent lifecycle, human/ai text messages, tool
# calls/results, errors -- each line labelled with the owning agent and its
# langgraph thread. Internal chain plumbing is not printed, and messages that
# propagate unchanged through several runs of the same agent+thread are
# deduplicated (emitted exactly once).
# --------------------------------------------------------------------------- #

TL_AGENT_COLORS = ("cyan", "magenta", "green", "yellow", "blue",
                   "bright_red", "bright_cyan", "bright_magenta")


def _agent_color(agent: str | None) -> str:
    # Deterministic (not python's randomized hash) so a given agent name
    # always gets the same color across server restarts.
    return TL_AGENT_COLORS[sum(ord(c) for c in (agent or "?")) % len(TL_AGENT_COLORS)]


def _tl_time(prefer: str, body: dict) -> str:
    """Timestamp for a timeline event: prefer the payload's own ISO time
    (batched events do not arrive in real time), else the arrival clock."""
    ts = _parse_iso_time(body.get(prefer))
    dt = datetime.fromtimestamp(ts) if ts is not None else datetime.now()
    return f"{dt:%H:%M:%S}.{dt.microsecond // 1000:03d}"


def _tl_print(when: str, agent: str | None, thread: str | None,
              markup: str) -> None:
    global TL_LABEL_WIDTH
    label = agent or "-"
    if thread:
        label = f"{label} ·{thread}"
    if len(label) > TL_LABEL_WIDTH:
        TL_LABEL_WIDTH = min(len(label), 42)
    console.print(
        f"[dim]\\[{when}][/dim] "
        f"[bold {_agent_color(agent)}]{label:<{TL_LABEL_WIDTH}}[/bold {_agent_color(agent)}] "
        f"{markup}"
    )


def _tl_msg_key(agent: str | None, thread: str | None, msg: dict) -> tuple[Any, ...]:
    """Identity of a chat message *within one agent+thread*: role + content +
    tool-call ids. Used to suppress the same artifact when it is propagated
    through several runs of the chain (or replayed as history by a later
    turn on the same thread)."""
    calls = tuple(sorted(
        (str(tc.get("name") or ""), str(tc.get("id") or ""))
        for tc in msg["tool_calls"] if isinstance(tc, dict)
    ))
    return (agent, thread, msg["role"], squash(msg["content"]), calls)


def _tl_msg_new(agent: str | None, thread: str | None, msg: dict) -> bool:
    key = _tl_msg_key(agent, thread, msg)
    if key in TIMELINE_SEEN:
        return False
    TIMELINE_SEEN.add(key)
    while len(TIMELINE_SEEN) > TIMELINE_SEEN_MAX:
        TIMELINE_SEEN.pop()
    return True


def timeline_run_start(run_id: str, name: str, run_type: str, inputs: Any,
                       body: dict, entry: dict) -> None:
    agent, thread = entry.get("agent"), entry.get("thread")
    label = agent or name
    when = _tl_time("start_time", body)

    if entry.get("boundary"):
        _tl_print(when, label, thread, f"[bold]▶ agent {name}[/bold]")

    # Human messages entering this agent's thread. Scanned from *every* run's
    # inputs (not just boundaries): the dedupe key makes propagation copies
    # (chain in -> llm in -> ...) -- and history replays on later turns of
    # the same thread -- print exactly once, whichever run carries the
    # message first.
    for msg in extract_messages(inputs):
        if msg["role"] != "human":
            continue
        if not _tl_msg_new(label, thread, msg):
            continue
        _tl_print(when, label, thread,
                  f"[{ROLE_STYLES['human']}]human[/]  {truncate(msg['content'], MAX_LEN)}")


def timeline_run_end(run_id: str, name: str, run_type: str, outputs: Any,
                     error: Any, duration_s: float | None, body: dict,
                     entry: dict) -> None:
    agent, thread = entry.get("agent"), entry.get("thread")
    label = agent or name
    when = _tl_time("end_time", body)
    dur = f" ({duration_s:.2f}s)" if duration_s is not None else ""

    if entry.get("boundary"):
        if error is not None:
            _tl_print(when, label, thread,
                      f"[bold red]✖ agent {name}{dur}[/bold red] "
                      f"[red]ERROR {truncate(squash(str(error)), MAX_LEN)}[/red]")
        else:
            _tl_print(when, label, thread,
                      f"[bold green]✔ agent {name}{dur}[/bold green]")
        return

    if error is not None:
        _tl_print(when, label, thread,
                  f"[bold red]✖ {name}[/bold red] "
                  f"[red]ERROR {truncate(squash(str(error)), MAX_LEN)}[/red]")
        return

    rt = (run_type or "").lower()
    if rt == "llm":
        for gen in extract_llm_generations(outputs):
            if not _tl_msg_new(label, thread, gen):
                continue
            text = squash(gen["content"])
            if text:
                _tl_print(when, label, thread,
                          f"[{ROLE_STYLES['ai']}]ai[/]     {truncate(text, MAX_LEN)}")
            for tc in gen["tool_calls"][:6]:
                if not isinstance(tc, dict):
                    continue
                _tl_print(when, label, thread,
                          f"[green]⚒ {tc.get('name', '?')}[/green]"
                          f"({compact_line(tc.get('args'), cap=320)})")
            if len(gen["tool_calls"]) > 6:
                _tl_print(when, label, thread,
                          f"[dim]  … +{len(gen['tool_calls']) - 6} more tool call(s)[/dim]")
    elif rt == "tool":
        args = entry.get("tool_args")
        call = f"{name}({compact_line(args, cap=320)})" if args is not None else name
        _tl_print(when, label, thread,
                  f"[green]tool[/green] {call} [dim]→[/dim] {_tool_result(outputs)}")


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #

app = FastAPI(title="mock-langsmith-tracer")


@app.middleware("http")
async def ignore_auth_and_log_unknown(request: Request, call_next):
    """Auth is not checked anywhere (no middleware actually enforces it);
    this middleware's job is simply to strip/ignore auth headers so that
    handlers never need to look at them, and to time requests.
    """
    # Explicitly drop any auth-ish headers from being relied upon downstream.
    request.scope["headers"] = [
        (k, v) for (k, v) in request.scope["headers"]
        if k.decode().lower() not in ("x-api-key", "authorization", "cookie")
    ]
    response = await call_next(request)
    return response


@app.get("/info")
async def info() -> JSONResponse:
    """Mock server capability banner. Required by the SDK's Client.info
    property, which is fetched lazily on first use.
    """
    return JSONResponse(
        {
            "version": "0.0.1-mock",
            "license_expiration_time": None,
            "batch_ingest_config": {
                "use_multipart_endpoint": False,
                "scale_up_qsize_trigger": 1000,
                "scale_up_nthreads_limit": 16,
                "scale_down_nempty_trigger": 4,
                "size_limit": 100,
                "size_limit_bytes": 20_971_520,
            },
            "instance_flags": {},
        }
    )


@app.post("/sessions")
async def create_session(request: Request) -> JSONResponse:
    """Mock "create project" endpoint. Just echoes back a dummy session
    (a.k.a. "project") record with a fresh id, which is all the SDK needs.
    """
    body = await _safe_json(request)
    session_id = str(body.get("id") or uuid.uuid4())
    name = body.get("name") or "default"

    if VIEW != "timeline":
        console.print(
            Panel(
                f"[bold]name[/bold]: {name}\n[bold]id[/bold]: [dim]{session_id}[/dim]",
                title="[bold white]📁 SESSION CREATED[/bold white]",
                border_style="white",
                expand=False,
            )
        )

    now = datetime.now(timezone.utc).isoformat()
    return JSONResponse(
        {
            "id": session_id,
            "name": name,
            "start_time": now,
            "end_time": None,
            "description": body.get("description"),
            "extra": body.get("extra"),
            "tenant_id": str(uuid.uuid4()),
            "reference_dataset_id": None,
        }
    )


def compute_depth(body: dict) -> int:
    """Nesting depth of a run. Preferred: count the dot-separated segments
    of ``dotted_order`` (each nesting level appends one segment). Fallback:
    walk the parent chain through the run registry."""
    dotted = body.get("dotted_order")
    if isinstance(dotted, str) and dotted:
        return dotted.count(".")
    parent_id = body.get("parent_run_id")
    depth = 0
    seen: set[str] = set()
    while parent_id and parent_id in RUN_REGISTRY and parent_id not in seen:
        seen.add(parent_id)
        depth += 1
        parent_id = RUN_REGISTRY[parent_id].get("parent_run_id")
    return depth


def _parse_iso_time(value: Any) -> float | None:
    """Parse an ISO-8601 timestamp into epoch seconds (None if absent or
    unparseable). Needed because batched runs do not arrive in real time,
    so durations must come from the payload, not from arrival clocks."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _run_metadata(body: dict) -> dict:
    """The run's ``extra.metadata`` dict (empty if absent/malformed)."""
    extra = body.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("metadata"), dict):
        return extra["metadata"]
    return {}


def _classify_agent(name: str, run_type: str, meta: dict,
                    parent_entry: dict | None,
                    depth: int) -> tuple[str | None, str | None, bool]:
    """Determine which *agent* and langgraph *thread* a run belongs to, and
    whether the run is an agent boundary (i.e. a named agent graph starting).

    Primary signal: langchain's ``lc_agent_name`` metadata (set by
    ``create_agent(name=...)`` and propagated to every run underneath it; a
    nested agent re-asserts its own name, so a *changed* value marks the
    subagent boundary). Fallback for graphs built without it (e.g.
    ``langgraph.prebuilt.create_react_agent``): a chain/agent run at nesting
    depth 0 is the root graph, and a chain/agent run whose direct parent is a
    *tool* run is a sub-agent graph invoked by that tool -- in both cases the
    run's own name is the agent's name (unless it is internal plumbing).

    Returns ``(agent, thread, is_boundary)``.
    """
    rt = (run_type or "").lower()
    graph_like = rt in ("chain", "agent")
    lc = meta.get("lc_agent_name")
    thread = meta.get("thread_id")
    parent_agent = parent_entry.get("agent") if parent_entry else None
    if thread is None:
        thread = parent_entry.get("thread") if parent_entry else None

    if isinstance(lc, str) and lc:
        agent = lc
        if parent_entry is not None:
            boundary = lc != parent_agent
        else:
            boundary = depth == 0 and graph_like
        return agent, thread, boundary

    agent = parent_agent
    boundary = False
    if graph_like and depth == 0:
        agent, boundary = name, True
    elif (graph_like and parent_entry is not None
          and parent_entry.get("run_type") == "tool"
          and name not in INTERNAL_GRAPH_NAMES):
        agent, boundary = name, True
    return agent, thread, boundary


def _register_run(run_id: str, name: str, run_type: str, depth: int,
                  body: dict, agent: str | None = None,
                  thread: str | None = None,
                  boundary: bool = False) -> dict[str, Any]:
    entry = {
        "name": name,
        "run_type": run_type,
        "start": time.perf_counter(),
        "start_time": _parse_iso_time(body.get("start_time")),
        "parent_run_id": body.get("parent_run_id"),
        "depth": depth,
        "agent": agent,
        "thread": thread,
        "boundary": boundary,
        # tool args are recalled at PATCH time to render "name(args) -> result"
        "tool_args": body.get("inputs") if (run_type or "").lower() == "tool" else None,
    }
    RUN_REGISTRY[run_id] = entry
    while len(RUN_REGISTRY) > RUN_REGISTRY_MAX:
        RUN_REGISTRY.pop(next(iter(RUN_REGISTRY)))
    return entry


def _run_duration(body: dict, entry: dict | None) -> float | None:
    """Duration of a run. Prefer the payload's own ISO start/end times
    (accurate even for batched/merged events); fall back to the arrival
    clock stored in the registry (classic per-run POST -> PATCH traffic)."""
    end = _parse_iso_time(body.get("end_time"))
    start = _parse_iso_time(body.get("start_time"))
    if start is None and entry:
        start = entry.get("start_time")
    if end is not None and start is not None:
        return max(0.0, end - start)
    if entry and "start" in entry:
        return time.perf_counter() - entry["start"]
    return None


def ingest_run(body: dict, operation: str) -> str:
    """Single source of truth for ingesting one run operation, shared by
    every transport endpoint (POST /runs, PATCH /runs/{id}, the entries of
    POST /runs/batch and the parts of POST /runs/multipart).

    ``operation == "post"``: a run *started*. With batched transports the
    SDK merges a run's patch into its post when both land in the same
    client-side batch (``combine_serialized_queue_operations``), so a post
    may already be a *finished* run (it then carries ``end_time`` and
    possibly ``outputs`` / ``error``) -- log START *and* END for it.

    ``operation == "patch"``: a run *finished* (successfully or not).
    Returns the run id.
    """
    run_id = str(body.get("id") or uuid.uuid4())

    if operation == "post":
        name = body.get("name") or "<unnamed>"
        run_type = body.get("run_type") or "chain"
        depth = compute_depth(body)
        meta = _run_metadata(body)
        parent_entry = (RUN_REGISTRY.get(body["parent_run_id"])
                        if body.get("parent_run_id") else None)
        agent, thread, boundary = _classify_agent(name, run_type, meta,
                                                  parent_entry, depth)
        entry = _register_run(run_id, name, run_type, depth, body,
                              agent=agent, thread=thread, boundary=boundary)
        extra = body.get("extra") if isinstance(body.get("extra"), dict) else None

        if VIEW == "timeline":
            timeline_run_start(run_id, name, run_type, body.get("inputs"),
                               body, entry)
        else:
            log_run_start(run_id, name, run_type, body.get("inputs"), extra,
                          depth, agent=agent, thread=thread)

        # A post that already carries end_time is a *complete* run: either the
        # SDK merged the run's patch into its post (both queued in the same
        # client-side batch), or the tracer deferred posting until the run
        # finished (langchain_core "defers_inputs" runs are only POSTed at
        # end). Log START *and* END for it. NOTE: fresh posts DO carry
        # ``outputs: {}`` (the RunTree default), so outputs alone must NOT
        # trigger the END -- end_time is the only reliable signal.
        if body.get("end_time") is not None:
            duration_s = _run_duration(body, entry)
            RUN_REGISTRY.pop(run_id, None)
            if VIEW == "timeline":
                timeline_run_end(run_id, name, run_type, body.get("outputs"),
                                 body.get("error"), duration_s, body, entry)
            else:
                log_run_end(run_id, name, run_type, body.get("outputs"),
                            body.get("error"), duration_s, depth,
                            agent=agent, thread=thread)
    else:  # patch
        entry = RUN_REGISTRY.pop(run_id, {})
        name = body.get("name") or entry.get("name") or "<unnamed>"
        run_type = body.get("run_type") or entry.get("run_type") or "chain"
        depth = entry.get("depth")
        if depth is None:
            depth = compute_depth(body)
        if "agent" not in entry:
            # Post never seen (or registry evicted): classify from the patch
            # body itself -- patches carry the same metadata as posts.
            meta = _run_metadata(body)
            parent_entry = (RUN_REGISTRY.get(body["parent_run_id"])
                            if body.get("parent_run_id") else None)
            entry.update(dict(zip(
                ("agent", "thread", "boundary"),
                _classify_agent(name, run_type, meta, parent_entry, depth))))
        if VIEW == "timeline":
            timeline_run_end(run_id, name, run_type, body.get("outputs"),
                             body.get("error"), _run_duration(body, entry),
                             body, entry)
        else:
            log_run_end(run_id, name, run_type, body.get("outputs"),
                        body.get("error"), _run_duration(body, entry), depth,
                        agent=entry.get("agent"), thread=entry.get("thread"))
    return run_id


@app.post("/runs")
async def create_run(request: Request) -> JSONResponse:
    """Fired when a chain / llm / tool / agent / retriever run *starts*
    (classic, per-run protocol)."""
    body = await _safe_json(request)
    run_id = ingest_run(body, "post")
    return JSONResponse({"id": run_id}, status_code=202)


@app.patch("/runs/{run_id}")
async def update_run(run_id: str, request: Request) -> JSONResponse:
    """Fired when a chain / llm / tool / agent / retriever run *ends*
    (successfully or with an error; classic, per-run protocol)."""
    body = await _safe_json(request)
    body.setdefault("id", run_id)
    run_id = ingest_run(body, "patch")
    return JSONResponse({"id": run_id}, status_code=202)


@app.post("/runs/batch")
async def batch_ingest(request: Request) -> JSONResponse:
    """Batched ingest (the SDK default, ``auto_batch_tracing=True``): the
    body is ``{"post": [<run>, ...], "patch": [<run>, ...]}`` where each
    entry is exactly the same run dict the classic endpoints would have
    received (patches additionally carry ``id`` in the body)."""
    body = await _safe_json(request)
    posts = body.get("post")
    patches = body.get("patch")
    posts = posts if isinstance(posts, list) else []
    patches = patches if isinstance(patches, list) else []

    if VIEW != "timeline":
        console.print(f"[dim]⇈ batch ingest: {len(posts)} post · "
                      f"{len(patches)} patch op(s)[/dim]")
    for op_body in posts:
        if isinstance(op_body, dict):
            ingest_run(op_body, "post")
    for op_body in patches:
        if isinstance(op_body, dict):
            ingest_run(op_body, "patch")
    return JSONResponse({}, status_code=200)


def _parse_multipart(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    """Minimal multipart/form-data parser -> [(part_name, part_payload)].
    (``python-multipart`` is deliberately not a dependency; the SDK's parts
    are plain ``name="..."`` sections with JSON payloads.)"""
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not m:
        return []
    delim = b"--" + m.group(1).encode()
    parts: list[tuple[str, bytes]] = []
    for section in body.split(delim):
        if section.startswith(b"\r\n"):
            section = section[2:]
        if section.endswith(b"\r\n"):
            section = section[:-2]
        if section in (b"", b"--"):
            continue
        header_blob, sep, payload = section.partition(b"\r\n\r\n")
        if not sep:
            continue
        for line in header_blob.split(b"\r\n"):
            if b"content-disposition" in line.lower():
                nm = re.search(rb'name="([^"]*)"', line)
                if nm:
                    parts.append((nm.group(1).decode(errors="replace"), payload))
                break
    return parts


def _reassemble_multipart_ops(parts: list[tuple[str, bytes]]) -> \
        tuple[list[dict], list[dict], int, int]:
    """Rebuild complete run dicts from ``post.<id>`` / ``patch.<id>`` parts.
    The main part (``post.<id>``) holds the run dict minus its big optional
    fields; each ``post.<id>.<field>`` part (inputs / outputs / events /
    extra / error / serialized) is JSON and merged back in. Returns
    (posts, patches, n_feedback, n_attachments) -- feedback and attachments
    are counted and dropped (not supported by this mock)."""
    ops: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    n_feedback = n_attachments = 0

    def _json(payload: bytes) -> Any:
        try:
            return json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            console.print("[dim red]  (malformed JSON part in multipart "
                          "body -- skipped)[/dim red]")
            return None

    for name, payload in parts:
        op, _, rest = name.partition(".")
        if op not in ("post", "patch"):
            if op == "feedback":
                n_feedback += 1
            elif op == "attachment":
                n_attachments += 1
            continue
        run_id, _, field = rest.partition(".")
        key = (op, run_id)
        if key not in ops:
            ops[key] = {}
            order.append(key)
        if field:
            value = _json(payload)
            if value is not None:
                ops[key][field] = value
        else:
            main = _json(payload)
            if isinstance(main, dict):
                ops[key].update({k: v for k, v in main.items() if v is not None})
            ops[key].setdefault("id", run_id)

    posts = [ops[k] for k in order if k[0] == "post"]
    patches = [ops[k] for k in order if k[0] == "patch"]
    return posts, patches, n_feedback, n_attachments


@app.post("/runs/multipart")
async def multipart_ingest(request: Request) -> JSONResponse:
    """Multipart/form-data variant of batched ingest (used when the server
    advertises ``use_multipart_endpoint: true`` or by explicit
    ``Client.multipart_ingest()`` calls). Same shared ingestion core."""
    if (request.headers.get("content-encoding") or "").lower() == "zstd":
        # Only ever used when /info advertises zstd compression, which this
        # mock never does. Accept-and-warn rather than fail the client.
        console.print(
            Panel(
                "[yellow]zstd-compressed multipart ingest is not supported by "
                "this mock (its /info never advertises it) -- body ignored.[/yellow]",
                title="[bold yellow]⚠ COMPRESSED MULTIPART IGNORED[/bold yellow]",
                border_style="yellow", expand=False,
            )
        )
        return JSONResponse({}, status_code=200)

    parts = _parse_multipart(request.headers.get("content-type", ""),
                             await request.body())
    posts, patches, n_feedback, n_attachments = _reassemble_multipart_ops(parts)

    extra_bits = []
    if n_feedback:
        extra_bits.append(f"{n_feedback} feedback")
    if n_attachments:
        extra_bits.append(f"{n_attachments} attachment(s)")
    suffix = f" (+{', '.join(extra_bits)} ignored)" if extra_bits else ""
    if VIEW != "timeline":
        console.print(f"[dim]⇈ multipart ingest: {len(posts)} post · "
                      f"{len(patches)} patch op(s){suffix}[/dim]")
    for op_body in posts:
        ingest_run(op_body, "post")
    for op_body in patches:
        ingest_run(op_body, "patch")
    return JSONResponse({}, status_code=200)


# --------------------------------------------------------------------------- #
# Catch-all fallback: anything else the SDK might probe for (feedback,
# datasets, ...) is accepted and ignored instead of 404ing/crashing the
# client, since those are out of scope for this mock. (Run ingest has its
# own real endpoints above: /runs, /runs/{id}, /runs/batch, /runs/multipart.)
# --------------------------------------------------------------------------- #

@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def catch_all(full_path: str, request: Request) -> JSONResponse:
    if VIEW != "timeline":
        console.print(f"[dim]  (ignored {request.method} /{full_path})[/dim]")
    return JSONResponse({}, status_code=200)


async def _safe_json(request: Request) -> dict:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        console.print(f"[dim red]  (malformed JSON body on {request.method} "
                      f"{request.url.path} -- treated as empty)[/dim red]")
        return {}


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal local mock server for LangSmith/LangFuse tracing (FastAPI + Rich)."
    )
    parser.add_argument(
        "--port", type=int, default=1984,
        help="Port to listen on (localhost only). Default: 1984",
    )
    parser.add_argument(
        "--max-len", "--max-message-length", dest="max_len", type=int, default=200,
        help="Max characters for any single string in logged payloads before truncation. Default: 200",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to. Default: 127.0.0.1 (localhost only)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Force console width (useful when stdout is redirected to a file, "
             "where rich would otherwise default to 80 columns). Default: auto",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Log full raw payloads (pre-semantic-rendering behaviour) instead of "
             "the compact chat-transcript view. Only applies to --view full.",
    )
    parser.add_argument(
        "--view", choices=("full", "timeline"), default="full",
        help="Log view. 'full' (default): every run as a nested, color-coded "
             "panel -- the whole chain, for debugging. 'timeline': a filtered, "
             "high-level chronological timeline of only agent lifecycle "
             "(start/end), human/LLM text messages, tool calls/results and "
             "errors -- one labelled line per event, with propagated message "
             "duplicates suppressed.",
    )
    return parser.parse_args()


def main() -> None:
    global MAX_LEN, RAW_PAYLOADS, VIEW, console
    args = parse_args()
    MAX_LEN = args.max_len
    RAW_PAYLOADS = args.raw
    VIEW = args.view
    if args.width:
        console = Console(width=args.width)

    console.rule("[bold green]Mock LangSmith Tracer[/bold green]")
    console.print(f"Listening on [bold]http://{args.host}:{args.port}[/bold]")
    console.print(f"Max message length: [bold]{MAX_LEN}[/bold] chars (truncated beyond that)")
    if VIEW == "timeline":
        console.print("Log view: [bold]timeline[/bold] (filtered: agent lifecycle, "
                      "human/ai messages, tool calls/results, errors)")
    else:
        console.print(f"Payload rendering: {'raw dumps' if RAW_PAYLOADS else 'compact (chat transcript)'}")
    console.print(
        "Point your SDK at this with e.g.:\n"
        f"  export LANGCHAIN_TRACING_V2=true\n"
        f"  export LANGCHAIN_ENDPOINT=http://{args.host}:{args.port}\n"
        f"  export LANGCHAIN_API_KEY=anything-goes  [dim](auth is ignored)[/dim]\n"
    )
    console.rule()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()