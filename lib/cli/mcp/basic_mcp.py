#!/usr/bin/env python3
"""
Guido MCP test server — tools designed to force real model→tool call cycles.

Tools:
  get_time        Returns the current UTC date and time.  The model cannot
                  know this without calling the tool, making it the simplest
                  forcing function for verifying tool-call flow end-to-end.

  calculate       Evaluates a math expression via Python's math module.
                  Tests argument passing and numeric result handling.
                  Only a safe subset of builtins is exposed.

  read_file       Reads a file from disk and returns its contents (truncated
                  to 4 KB).  Tests string argument passing and multi-line
                  result handling.

  echo            Returns its input unchanged.  Trivial round-trip test —
                  useful when debugging the TOOL_CALL parsing / dispatch loop
                  without caring about the actual result.

Wire format: JSON-RPC 2.0 over stdin/stdout (MCP stdio transport).
"""
import sys
import json
import math
import datetime
import os


# ── helpers ────────────────────────────────────────────────────────────────────

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def ok(msg_id, text):
    send({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        },
    })


def err(msg_id, text):
    send({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [{"type": "text", "text": f"Error: {text}"}],
            "isError": True,
        },
    })


# ── tool implementations ───────────────────────────────────────────────────────

def tool_get_time(_args):
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("UTC %Y-%m-%d %H:%M:%S")


# Safe builtins for calculate — no __import__, open, exec, eval, etc.
_SAFE_GLOBALS = {
    "__builtins__": {},
    **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
}


def tool_calculate(args):
    expr = str(args.get("expression", "")).strip()
    if not expr:
        raise ValueError("expression is required")
    if len(expr) > 500:
        raise ValueError("expression too long (max 500 chars)")
    # Rough guard: reject obviously dangerous tokens
    for bad in ("import", "exec", "eval", "open", "os", "sys", "__"):
        if bad in expr:
            raise ValueError(f"disallowed token: {bad!r}")
    result = eval(expr, _SAFE_GLOBALS)  # noqa: S307 — intentional, sandboxed
    return str(result)


def tool_read_file(args):
    path = str(args.get("path", "")).strip()
    if not path:
        raise ValueError("path is required")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path}")
    if os.path.isdir(path):
        entries = os.listdir(path)
        return f"Directory listing for {path}:\n" + "\n".join(sorted(entries))
    with open(path, "r", errors="replace") as fh:
        data = fh.read(4096)
    truncated = os.path.getsize(path) > 4096
    if truncated:
        data += "\n... (truncated)"
    return data


def tool_echo(args):
    return str(args.get("message", ""))


DISPATCH = {
    "get_time":   tool_get_time,
    "calculate":  tool_calculate,
    "read_file":  tool_read_file,
    "echo":       tool_echo,
}

# ── MCP tool manifest ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_time",
        "description": (
            "Return the current UTC date and time. "
            "Use this whenever the user asks what time or date it is."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Supports Python math operators and the full math module "
            "(sin, cos, sqrt, log, pi, e, etc.)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 ** 32' or 'sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file (or list a directory) and return "
            "the result as text. Paths starting with ~ are expanded. "
            "File reads are capped at 4 KB."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or ~ path to the file or directory",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "echo",
        "description": (
            "Return the message unchanged. "
            "Useful for testing the tool-call round-trip without side effects."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Text to echo back",
                }
            },
            "required": ["message"],
        },
    },
]


# ── main loop ──────────────────────────────────────────────────────────────────

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue

    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "guido-test-server", "version": "0.2.0"},
            },
        })

    elif method == "notifications/initialized":
        pass  # no response for notifications

    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}})

    elif method == "tools/call":
        params = msg.get("params", {})
        name   = params.get("name", "")
        args   = params.get("arguments") or {}

        fn = DISPATCH.get(name)
        if fn is None:
            err(msg_id, f"unknown tool: {name!r}")
            continue

        try:
            text = fn(args)
            ok(msg_id, text)
        except Exception as exc:
            err(msg_id, str(exc))
