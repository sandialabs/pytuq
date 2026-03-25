# MCP client with multi-provider support (Anthropic, OpenAI, Google Gemini).
# See https://modelcontextprotocol.info/docs/quickstart/client/
#
# Provider is auto-detected from the model name:
#   claude-*          → Anthropic  (requires ANTHROPIC_API_KEY)
#   gpt-* / o1* / o3* → OpenAI    (requires OPENAI_API_KEY)
#   gemini-*          → Google    (requires GOOGLE_API_KEY)
#
# Usage:
#   python client.py server.py                          # default Claude model
#   python client.py server.py --model gpt-4o
#   python client.py server.py --model gemini-2.5-flash
import os
import re
import time
import json
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(model: str) -> str:
    """Auto-detect the LLM provider from the model name."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    if m.startswith(("meta-llama", "meta_llama")):
        return "shirty"
    raise ValueError(
        f"Cannot auto-detect provider from model name '{model}'. "
        "Model names should start with 'claude-', 'gpt-'/'o1'/'o3', or 'gemini-'."
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _tool_result_to_str(tool_result) -> str:
    """Serialize an MCP tool result to a plain string."""
    if hasattr(tool_result, "content"):
        parts = []
        for block in tool_result.content:
            parts.append(block.text if hasattr(block, "text") else str(block))
        return "\n".join(parts)
    return str(tool_result)


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

class MCPClient:
    """MCP client that supports Anthropic, OpenAI, and Google Gemini models.

    The provider is auto-detected from the model name passed to the constructor.
    All three public methods (process_query, run_agent_loop, chat_loop) work
    transparently regardless of provider.
    """

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.provider = _detect_provider(model)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Ensure shirty-related attributes exist even if shirty import fails
        self._shirty_wrapper = False
        self._shirty_base = os.getenv("SHIRTY_API_BASE", "https://shirty.sandia.gov/api/v1")
        self._client = self._init_client()
        # Accumulates every tool call made during run_agent_loop / process_query.
        # Each entry: {"name": str, "args": dict, "result": dict | str}
        # "result" is the parsed JSON dict when the tool returns JSON, otherwise raw text.
        self.tool_results: list = []

    # ------------------------------------------------------------------
    # Client initialisation (provider-specific)
    # ------------------------------------------------------------------

    # Map each provider to the env var it requires.
    _REQUIRED_ENV = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
        "shirty":    "SHIRTY_API_KEY",
    }

    def _check_api_key(self):
        """Exit with a clear message if the required API-key env var is absent."""
        import sys
        var = self._REQUIRED_ENV.get(self.provider)
        if var and not os.getenv(var):
            sys.exit(
                f"Error: environment variable '{var}' is not set.\n"
                f"Provider '{self.provider}' requires this key to be present "
                f"(e.g. in a .env file or exported in your shell)."
            )

    def _init_client(self):
        self._check_api_key()
        if self.provider == "anthropic":
            import httpx
            from anthropic import Anthropic
            verify = os.getenv("MCP_CA_BUNDLE") or os.getenv("SSL_CERT_FILE") or True
            return Anthropic(http_client=httpx.Client(verify=verify))
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        if self.provider == "google":
            from google import genai
            return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if self.provider == "shirty":
            try:
                from shirty.client import ShirtyClient
                self._shirty_wrapper = True
                return ShirtyClient(api_key=os.getenv("SHIRTY_API_KEY"), base_url=os.getenv("SHIRTY_API_BASE", "https://shirty.sandia.gov/api/v1"))
            except Exception:
                import httpx
                self._shirty_wrapper = False
                self._shirty_base = os.getenv("SHIRTY_API_BASE", "https://shirty.sandia.gov/api/v1")
                return httpx.Client()

    # ------------------------------------------------------------------
    # Tool format conversion  (MCP → provider schema)
    # ------------------------------------------------------------------

    def _format_tools(self, mcp_tools) -> list:
        """Convert MCP tool definitions to the format expected by the provider."""
        if self.provider == "anthropic":
            return [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in mcp_tools
            ]
        if self.provider == "openai":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in mcp_tools
            ]
        if self.provider == "shirty":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in mcp_tools
            ]
        if self.provider == "google":
            # Gemini's Schema subset (OpenAPI 3.0 subset) does not support:
            #   - "$schema", "title", "default" (Pydantic v2 adds these everywhere)
            #   - "anyOf"/"oneOf" unions (Pydantic generates these for Optional types)
            # Stripping unsupported keys and flattening nullable unions prevents
            # MALFORMED_FUNCTION_CALL finish reasons on the very first turn.
            _STRIP = {"$schema", "title", "default"}

            def _clean(schema):
                if not isinstance(schema, dict):
                    return schema
                # Drop keys Gemini doesn't understand
                schema = {k: _clean(v) for k, v in schema.items() if k not in _STRIP}
                # Flatten anyOf/oneOf that are just a nullable wrapper around one type
                # e.g. pad_width: int = None  →  anyOf: [{"type":"integer"},{"type":"null"}]
                for key in ("anyOf", "oneOf"):
                    if key in schema:
                        non_null = [s for s in schema[key] if s.get("type") != "null"]
                        if len(non_null) == 1:
                            merged = {k: v for k, v in schema.items() if k != key}
                            merged.update(_clean(non_null[0]))
                            return merged
                return schema
            return [
                {"name": t.name, "description": t.description, "parameters": _clean(t.inputSchema)}
                for t in mcp_tools
            ]

    # ------------------------------------------------------------------
    # LLM call → normalised result
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list, tools: list, max_tokens: int = 1000, max_retries: int = 6):
        """Call the configured LLM, retrying automatically on rate-limit (429) errors.

        Returns:
            text_parts  : list[str]  – text blocks from the response
            tool_calls  : list[dict] – each dict has keys: id, name, args, _raw
            raw_response: the raw SDK response object
        """
        for attempt in range(max_retries):
            try:
                if self.provider == "anthropic":
                    return self._call_anthropic(messages, tools, max_tokens)
                if self.provider == "openai":
                    return self._call_openai(messages, tools, max_tokens)
                if self.provider == "shirty":
                    return self._call_shirty(messages, tools, max_tokens)
                if self.provider == "google":
                    return self._call_google(messages, tools, max_tokens)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    getattr(e, "status_code", None) == 429
                    or "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "RateLimitError" in type(e).__name__
                )
                if is_rate_limit and attempt < max_retries - 1:
                    # Parse suggested retry delay from the error message if present.
                    match = re.search(r"retry in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                    delay = float(match.group(1)) + 1.0 if match else 15.0
                    print(f"  Rate limited (429). Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries - 1}...")
                    time.sleep(delay)
                else:
                    raise

    def _call_anthropic(self, messages, tools, max_tokens):
        raw = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
        )
        text_parts, tool_calls = [], []
        for block in raw.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "args": block.input, "_raw": block})
        return text_parts, tool_calls, raw

    def _call_openai(self, messages, tools, max_tokens):
        raw = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )
        msg = raw.choices[0].message
        text_parts = [msg.content] if msg.content else []
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "_raw": tc,
                })
        return text_parts, tool_calls, raw

    def _call_google(self, messages, tools, max_tokens):
        from google.genai import types
        config_kwargs: dict = {"max_output_tokens": max_tokens}
        if tools:
            config_kwargs["tools"] = [types.Tool(function_declarations=tools)]
        raw = self._client.models.generate_content(
            model=self.model,
            contents=messages,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Guard: prompt-level safety block returns an empty candidates list.
        if not raw.candidates:
            feedback = getattr(raw, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", "unknown") if feedback else "unknown"
            raise RuntimeError(f"Gemini returned no candidates (blocked: {reason})")

        candidate = raw.candidates[0]

        # Log non-normal finish reasons so the user can see what happened.
        finish = getattr(candidate, "finish_reason", None)
        if finish and str(finish) not in ("FinishReason.STOP", "1"):
            print(f"  [Gemini] finish_reason={finish}")

        # Guard: content can be None for safety-filtered responses.
        if candidate.content is None:
            raise RuntimeError(f"Gemini candidate has no content (finish_reason={finish})")

        text_parts, tool_calls = [], []
        for part in candidate.content.parts:
            # Check function_call first — on some SDK versions part.text is ""
            # (not None) even when the part is purely a function call.
            if part.function_call is not None:
                fc = part.function_call
                tool_calls.append({
                    "id": f"{fc.name}_{id(fc)}",
                    "name": fc.name,
                    "args": dict(fc.args),
                    "_raw": fc,
                })
            elif part.text:  # truthy: skips None and empty string
                text_parts.append(part.text)
        return text_parts, tool_calls, raw


    def _call_shirty(self, messages, tools, max_tokens):
        """Call the Shirty endpoint (wrapper or HTTP fallback)."""
        import json
        if getattr(self, "_shirty_wrapper", False):
            raw = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                tools=tools if tools else None,
            )
            msg = raw.choices[0].message
            text_parts = [msg.content] if getattr(msg, "content", None) else []
            tool_calls = []
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                        "_raw": tc,
                    })
            return text_parts, tool_calls, raw
        else:
            import httpx
            url = f"{self._shirty_base.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {os.getenv('SHIRTY_API_KEY')}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
            if tools:
                payload["tools"] = tools
            resp = self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            raw = resp.json()
            msg = raw["choices"][0]["message"]
            text_parts = [msg.get("content")] if msg.get("content") else []
            tool_calls = []
            for tc in msg.get("tool_calls", []) or []:
                tool_calls.append({
                    "id": tc.get("id") or f"{tc.get('function', {}).get('name')}_{id(tc)}",
                    "name": tc.get("function", {}).get("name"),
                    "args": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    "_raw": tc,
                })
            return text_parts, tool_calls, raw

    # ------------------------------------------------------------------
    # Message history helpers (provider-specific)
    # ------------------------------------------------------------------

    def _make_user_message(self, content: str) -> dict:
        if self.provider == "google":
            return {"role": "user", "parts": [{"text": content}]}
        return {"role": "user", "content": content}

    def _make_assistant_message(self, raw_response) -> object:
        """Return the assistant turn in the format the provider expects back."""
        if self.provider == "anthropic":
            return {"role": "assistant", "content": raw_response.content}
        if self.provider == "openai":
            msg = raw_response.choices[0].message
            return {"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls}
        if self.provider == "shirty":
            # Shirty responses mirror the OpenAI-style structure for both
            # the SDK wrapper and the JSON HTTP fallback. Normalize both.
            try:
                # SDK-like object path
                msg = raw_response.choices[0].message
                return {"role": "assistant", "content": getattr(msg, "content", None), "tool_calls": getattr(msg, "tool_calls", None)}
            except Exception:
                # Fallback JSON path
                msg = raw_response["choices"][0]["message"]
                return {"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls")}
        if self.provider == "google":
            # Return the Content object directly; the SDK accepts it back as-is.
            return raw_response.candidates[0].content

    def _make_tool_result_messages(self, tool_calls: list, results_map: dict) -> list:
        """Build message(s) that deliver tool results back to the LLM."""
        if self.provider == "anthropic":
            return [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": results_map[tc["id"]].content,
                    }
                    for tc in tool_calls
                ],
            }]
        if self.provider == "openai":
            return [
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _tool_result_to_str(results_map[tc["id"]]),
                }
                for tc in tool_calls
            ]
        if self.provider == "shirty":
            return [
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _tool_result_to_str(results_map[tc["id"]]),
                }
                for tc in tool_calls
            ]
        if self.provider == "google":
            parts = [
                {
                    "function_response": {
                        "name": tc["name"],
                        "response": {"result": _tool_result_to_str(results_map[tc["id"]])},
                    }
                }
                for tc in tool_calls
            ]
            return [{"role": "user", "parts": parts}]

    # ------------------------------------------------------------------
    # Server connection
    # ------------------------------------------------------------------

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        response = await self.session.list_tools()
        print(f"\nConnected to server with tools: {[t.name for t in response.tools]}")
        print(f"Using model : {self.model}")
        print(f"Provider    : {self.provider}")

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    # Sentinel sent to the model after a text-only turn to nudge it to call a tool.
    _CONTINUE_PROMPT = "Please continue with the next step and call the appropriate tool."

    async def run_agent_loop(
        self,
        user_prompt: str,
        max_turns: int = 20,
        model: str = None,
        max_tokens: int = 8192,
    ) -> str:
        """Run an agentic LLM ↔ tool loop until no tool calls are returned.

        Args:
            user_prompt: The initial user message.
            max_turns: Maximum LLM ↔ tool iterations before stopping.
            model: Override the model for this call (must be same provider).
            max_tokens: Max output tokens per LLM call.

        Returns:
            All text produced by the assistant across all turns.

        Raises:
            Any exception from the LLM call that exhausts all retries (e.g. daily
            quota exceeded). The caller's try/except will surface this properly.
        """
        if model and model != self.model:
            self.model = model

        self.tool_results = []   # reset for this run
        messages = [self._make_user_message(user_prompt)]

        resp = await self.session.list_tools()
        available_tools = self._format_tools(resp.tools)

        final_text_parts: list[str] = []
        text_only_turns = 0  # consecutive turns with text but no tool calls
        MAX_TEXT_ONLY = 3    # give the model a few chances to produce a tool call

        for turn in range(max_turns):
            print(f"\n--- Agent Turn {turn + 1} ---")
            # Let exceptions propagate — silently breaking here caused the loop to
            # return "" (displayed as "none" by the caller) even when a real error
            # (e.g. daily quota exhaustion) was the culprit.
            text_parts, tool_calls, raw_resp = self._call_llm(messages, available_tools, max_tokens)

            final_text_parts.extend(text_parts)

            if not tool_calls:
                if text_parts and text_only_turns < MAX_TEXT_ONLY:
                    # Gemini often returns a text-only "planning" turn before
                    # calling any tools.  Keep going instead of terminating.
                    text_only_turns += 1
                    print(f"  (text-only turn {text_only_turns}/{MAX_TEXT_ONLY}, prompting to continue)")
                    messages.append(self._make_assistant_message(raw_resp))
                    messages.append(self._make_user_message(self._CONTINUE_PROMPT))
                    continue
                print("Agent returned no tool_use; finishing.")
                break

            text_only_turns = 0  # reset whenever the model uses a tool

            # Execute all tool calls
            results_map = {}
            for tc in tool_calls:
                print(f"Invoking tool: {tc['name']} with args: {tc['args']}")
                try:
                    result = await self.session.call_tool(tc["name"], tc["args"])
                except Exception as e:
                    print(f"Tool {tc['name']} raised exception: {e}")
                    result = type("ErrResult", (), {"content": [type("T", (), {"text": str(e)})()]})()
                results_map[tc["id"]] = result

                # Record the tool call and its result for inspection after the loop.
                raw_text = _tool_result_to_str(result)
                try:
                    parsed = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    parsed = raw_text
                self.tool_results.append({"name": tc["name"], "args": tc["args"], "result": parsed})

                if tc["name"] == "task_finished":
                    # Capture the summary the agent provided — Gemini in particular
                    # returns no free text alongside tool calls, so this is often the
                    # only meaningful text in the entire loop.
                    summary = tc["args"].get("summary", "")
                    if summary and summary not in final_text_parts:
                        final_text_parts.append(summary)
                    print("task_finished invoked by agent; ending loop.")
                    return "\n".join(final_text_parts)

            messages.append(self._make_assistant_message(raw_resp))
            messages.extend(self._make_tool_result_messages(tool_calls, results_map))

        else:
            print("Max turns reached; ending agent loop.")

        return "\n".join(final_text_parts)

    # ------------------------------------------------------------------
    # process_query  (single-pass with optional tool use)
    # ------------------------------------------------------------------

    async def process_query(self, query: str) -> str:
        """Process a single query, executing any tool calls the model requests."""
        messages = [self._make_user_message(query)]
        resp = await self.session.list_tools()
        available_tools = self._format_tools(resp.tools)

        print(f"Making initial {self.provider} API call...")
        try:
            text_parts, tool_calls, raw_resp = self._call_llm(messages, available_tools)
        except Exception as e:
            import traceback
            print(f"\nError in initial API call:\n  {type(e).__name__}: {e}")
            if hasattr(e, "response"):
                print(f"  HTTP Status : {e.response.status_code}")
                print(f"  Response    : {e.response.text}")
            traceback.print_exc()
            raise

        print("After initial API call...")
        final_text = list(text_parts)

        for tc in tool_calls:
            final_text.append(f"[Calling tool {tc['name']} with args {tc['args']}]")
            result = await self.session.call_tool(tc["name"], tc["args"])

            messages.append(self._make_assistant_message(raw_resp))
            messages.extend(self._make_tool_result_messages([tc], {tc["id"]: result}))

            text_parts2, _tc2, raw_resp = self._call_llm(messages, available_tools)
            final_text.extend(text_parts2)

        return "\n".join(final_text)

    # ------------------------------------------------------------------
    # Interactive chat loop
    # ------------------------------------------------------------------

    async def chat_loop(self):
        """Run an interactive chat loop.

        Preserves the original interactive behaviour: read a line from stdin,
        send it through process_query, and print the response.
        """
        print("\nMCP Client Started!")
        print(f"Model: {self.model}  |  Provider: {self.provider}")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Clean up resources."""
        try:
            if getattr(self, "exit_stack", None) is None:
                return
            await self.exit_stack.aclose()
        except Exception as e:
            import traceback
            print(f"\nWarning: cleanup error: {type(e).__name__}: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="MCP Client – supports Anthropic, OpenAI, and Google Gemini models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py server.py
  python client.py server.py --model gpt-4o
  python client.py server.py --model gemini-2.0-flash

Provider is auto-detected from the model name prefix:
  claude-*           → Anthropic  (ANTHROPIC_API_KEY)
  gpt-* / o1* / o3*  → OpenAI    (OPENAI_API_KEY)
  gemini-*           → Google    (GOOGLE_API_KEY)
""",
    )
    parser.add_argument("server_script", help="Path to the MCP server script (.py or .js)")
    parser.add_argument(
        "--model",
        default=MCPClient.DEFAULT_MODEL,
        help="Model name to use (default: %(default)s)",
    )
    args = parser.parse_args()

    client = MCPClient(model=args.model)
    try:
        await client.connect_to_server(args.server_script)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
