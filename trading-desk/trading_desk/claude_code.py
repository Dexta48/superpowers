"""Run the agents through Claude Code instead of the API, so they use a Claude
subscription (Pro/Max) rather than pay-per-call billing. Needs the `claude`
command installed and logged in on this computer.

`ClaudeCodeClient` looks like the slice of the Anthropic client the agents use
(`client.beta.messages.stream(...)`), and translates each call into one
headless `claude -p` run."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from types import SimpleNamespace as NS

INSTALL_HINT = (
    "Claude Code was not found. Install it (https://claude.com/claude-code), run `claude` once "
    "to log in with your Claude account, then try again."
)


class _Run:
    def __init__(self, system: str, prompt: str, search: bool, schema):
        self.schema, self.final = schema, None
        cmd = [
            "claude", "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
            "--system-prompt", system,
        ]
        # Only web search for the news desk; no tools at all for the other two.
        cmd += ["--tools", "WebSearch", "--allowedTools", "WebSearch"] if search else ["--tools", ""]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema.model_json_schema())]
        # Model and effort follow your Claude Code defaults unless set explicitly.
        if os.environ.get("TRADING_DESK_MODEL"):
            cmd += ["--model", os.environ["TRADING_DESK_MODEL"]]
        if os.environ.get("TRADING_DESK_EFFORT"):
            cmd += ["--effort", os.environ["TRADING_DESK_EFFORT"]]
        self.cwd = tempfile.TemporaryDirectory()  # empty folder: no project files or instructions leak in
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=self.cwd.name, text=True, encoding="utf-8",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.cwd.cleanup()
            raise RuntimeError(INSTALL_HINT) from None
        self.proc.stdin.write(prompt)
        self.proc.stdin.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.cwd.cleanup()
        return False

    def __iter__(self):
        """Yield API-style stream events; remember the result line."""
        as_text = False  # inside the structured-output block, whose JSON is the agent's writing
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("type") == "result":
                self.final = msg
            if msg.get("type") != "stream_event":
                continue
            ev = msg["event"]
            if ev["type"] == "content_block_start":
                block = ev["content_block"]
                kind = block["type"]
                if kind == "tool_use":
                    as_text = block.get("name") != "WebSearch"
                    kind = "text" if as_text else "server_tool_use"
                else:
                    as_text = False
                yield NS(type="content_block_start", content_block=NS(type=kind))
            elif ev["type"] == "content_block_delta":
                d = ev["delta"]
                if d["type"] == "input_json_delta" and as_text:
                    yield NS(type="content_block_delta", delta=NS(type="text_delta", text=d["partial_json"]))
                elif d["type"] in ("text_delta", "thinking_delta", "input_json_delta"):
                    yield NS(type="content_block_delta", delta=NS(**d))
            elif ev["type"] == "content_block_stop":
                yield NS(type="content_block_stop")

    def get_final_message(self):
        for _ in self:  # drain, in case the caller stopped early
            pass
        code = self.proc.wait()
        f = self.final
        if f is None or f.get("is_error") or code != 0:
            detail = (f or {}).get("result") or self.proc.stderr.read().strip() or f"exit code {code}"
            raise RuntimeError(f"Claude Code run failed: {detail}")
        text = f.get("result") or ""
        parsed = None
        if self.schema is not None:
            data = f.get("structured_output")
            parsed = self.schema.model_validate(data if data is not None else json.loads(text))
        return NS(stop_reason="end_turn", stop_details=None, content=[NS(type="text", text=text)], parsed_output=parsed)


class ClaudeCodeClient:
    def __init__(self):
        self.beta = NS(messages=self)

    def stream(self, *, system, messages, tools=None, output_format=None, **_ignored):
        return _Run(system, messages[0]["content"], bool(tools), output_format)
