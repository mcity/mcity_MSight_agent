"""LLM provider clients behind one interface: chat(messages, tools, tool_choice)
takes OpenAI-format messages and returns an object with .content and .tool_calls."""
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from groq import AsyncGroq
import google.generativeai as genai
from google.generativeai import protos
from google.generativeai.types import content_types


class _FakeFunction:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    __slots__ = ("id", "function")

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    __slots__ = ("content", "tool_calls")

    def __init__(self, content: Optional[str], tool_calls: list):
        self.content = content
        self.tool_calls = tool_calls


def _tool_call_fields(tc) -> tuple[str, str, dict]:
    """(id, name, parsed arguments) from an OpenAI-format tool call, dict or object."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        tc_id, name, raw_args = tc.get("id", ""), fn.get("name", ""), fn.get("arguments", "{}")
    else:
        fn = getattr(tc, "function", None)
        tc_id = getattr(tc, "id", "")
        name, raw_args = (getattr(fn, "name", ""), getattr(fn, "arguments", "{}")) if fn else ("", "{}")
    try:
        args = json.loads(raw_args)
    except (TypeError, ValueError):
        args = {}
    return tc_id, name, args


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, messages, tools: Optional[List] = None, tool_choice: Optional[str] = None):
        raise NotImplementedError

    async def summarize_classification_report(self, tool_output: str) -> str:
        prompt = f"""
        Here's a classification report/results from an object detection model run. Briefly summarize how the model performed.

        {tool_output}

        Only mention:
        - Which class did best
        - Which class was worst
        - What does the micro/macro F1 tell us about generalization
        """
        return await self._summarize(prompt)

    async def _summarize(self, prompt: str) -> str:
        response = await self.chat([{"role": "user", "content": prompt}])
        return getattr(response, "content", "") or ""


class _OpenAICompatibleClient(BaseLLMClient):
    """OpenAI and Groq share the chat-completions API."""
    client = None
    model = ""

    def _supports_temperature(self) -> bool:
        return True

    async def chat(self, messages, tools=None, tool_choice=None):
        kwargs = dict(model=self.model, messages=messages)
        if self._supports_temperature():
            kwargs["temperature"] = 0.1
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = False  # prevents missing-field errors with tool_choice="required"
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message


class OpenAIClient(_OpenAICompatibleClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _supports_temperature(self) -> bool:
        return not self.model.startswith(("gpt-5", "o1", "o3", "o4"))


class GroqClient(_OpenAICompatibleClient):
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        # llama3-70b-8192 (former default) was decommissioned by Groq on 2025-05-31.
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class GeminiClient(BaseLLMClient):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # gemini-1.5-flash has been retired.
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    async def chat(self, messages, tools=None, tool_choice=None):
        system_text, contents = self._to_gemini_contents(messages)
        # system_instruction is constructor-only, and system text varies per turn.
        model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system_text or None)

        kwargs = {"generation_config": {"temperature": 0.1}}
        if tools:
            kwargs["tools"] = [self._convert_tools(tools)]
            kwargs["tool_config"] = {
                "function_calling_config": {"mode": "any" if tool_choice == "required" else "auto"}
            }

        try:
            response = await model.generate_content_async(contents, **kwargs)
        except Exception as e:
            logging.warning(f"[GEMINI] API error: {e}")
            raise

        return self._to_fake_message(response)

    async def _summarize(self, prompt: str) -> str:
        model = genai.GenerativeModel(model_name=self.model_name)
        response = await model.generate_content_async(prompt, generation_config={"temperature": 0.1})
        return response.text.strip()

    @staticmethod
    def _convert_tools(tools: list) -> dict:
        decls = []
        for t in tools:
            fn = t.get("function", {})
            params = fn.get("parameters", {"type": "object", "properties": {}})
            # Gemini's Schema has no additionalProperties; strip it.
            params = json.loads(json.dumps(params))  # deep copy; strip mutates in place
            content_types.strip_additional_properties(params)
            decls.append({"name": fn.get("name", ""), "description": fn.get("description", ""), "parameters": params})
        return {"function_declarations": decls}

    @staticmethod
    def _to_gemini_contents(openai_messages: list) -> tuple[str, list]:
        """OpenAI-format messages -> Gemini contents (tool results matched by name; no ids)."""
        system_parts: list[str] = []
        result: list[dict] = []

        def _append_part(role: str, part):
            if result and result[-1]["role"] == role:
                result[-1]["parts"].append(part)
            else:
                result.append({"role": role, "parts": [part]})

        for msg in openai_messages:
            role = msg.get("role", "")

            if role == "system":
                text = (msg.get("content", "") or "").strip()
                if text:
                    system_parts.append(text)
                continue

            if role == "tool":
                fr_part = protos.Part(function_response=protos.FunctionResponse(
                    name=msg.get("name", ""),
                    response={"result": str(msg.get("content", ""))},
                ))
                _append_part("user", fr_part)
                continue

            if role == "assistant":
                text = msg.get("content") or ""
                if text:
                    _append_part("model", {"text": text})
                for tc in msg.get("tool_calls") or []:
                    _, fn_name, args = _tool_call_fields(tc)
                    _append_part("model", protos.Part(function_call=protos.FunctionCall(name=fn_name, args=args)))
                continue

            if role == "user":
                content = msg.get("content", "") or ""
                if content:
                    _append_part("user", {"text": content})
                continue

        while result and result[0]["role"] != "user":
            result.pop(0)

        return "\n\n".join(system_parts), result

    @staticmethod
    def _to_fake_message(response) -> "_FakeMessage":
        text_parts: list[str] = []
        tool_calls: list[_FakeToolCall] = []

        parts = response.candidates[0].content.parts if response.candidates else []
        for i, part in enumerate(parts):
            if "text" in part and part.text:
                text_parts.append(part.text)
            elif "function_call" in part:
                fc = part.function_call
                # args is a protobuf Struct -- numbers round-trip as float (5 -> 5.0).
                args = {
                    k: (int(v) if isinstance(v, float) and v.is_integer() else v)
                    for k, v in dict(fc.args).items()
                }
                tool_calls.append(_FakeToolCall(
                    id=f"gemini_{fc.name}_{i}_{uuid.uuid4().hex[:8]}",
                    name=fc.name,
                    arguments=json.dumps(args),
                ))

        return _FakeMessage(content="".join(text_parts) or None, tool_calls=tool_calls)


class ClaudeClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    async def chat(self, messages, tools=None, tool_choice=None):
        system_parts, anthropic_messages = self._to_anthropic_messages(messages)

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": "Hello"}]

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
            "temperature": 0.1,
        }

        if system_parts:
            # Only cache the first (static) block; dynamic injections don't get cache_control.
            system_blocks = [
                {"type": "text", "text": system_parts[0], "cache_control": {"type": "ephemeral"}},
                *[{"type": "text", "text": part} for part in system_parts[1:]],
            ]
            kwargs["system"] = system_blocks

        if tools:
            converted = self._convert_tools(tools)
            if converted:
                converted[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = converted
            kwargs["tool_choice"] = {"type": "any"} if tool_choice == "required" else {"type": "auto"}

        try:
            response = await self.client.messages.create(**kwargs)
        except Exception as e:
            logging.warning(f"[CLAUDE] API error: {e}")
            raise

        return self._to_fake_message(response)

    @staticmethod
    def _convert_tools(tools: list) -> list:
        result = []
        for t in tools:
            fn = t.get("function", {})
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    @staticmethod
    def _to_anthropic_messages(openai_messages: list) -> tuple[str, list]:
        """Convert OpenAI-format messages to Anthropic format (system list + messages list)."""
        system_parts: list[str] = []
        result: list[dict] = []

        for msg in openai_messages:
            role = msg.get("role", "")

            if role == "system":
                text = (msg.get("content", "") or "").strip()
                if text:
                    system_parts.append(text)
                continue

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": str(msg.get("content", "")),
                }
                if result and result[-1]["role"] == "user":
                    prev = result[-1]["content"]
                    if isinstance(prev, list):
                        prev.append(block)
                    else:
                        result[-1]["content"] = ([{"type": "text", "text": prev}] if prev else []) + [block]
                else:
                    result.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                tool_calls_raw = msg.get("tool_calls") or []
                blocks: list[dict] = []

                text = msg.get("content") or ""
                if text:
                    blocks.append({"type": "text", "text": text})

                for tc in tool_calls_raw:
                    tc_id, fn_name, inp = _tool_call_fields(tc)
                    blocks.append({"type": "tool_use", "id": tc_id, "name": fn_name, "input": inp})

                if not blocks:
                    blocks = [{"type": "text", "text": " "}]

                result.append({"role": "assistant", "content": blocks})
                continue

            if role == "user":
                content = msg.get("content", "") or ""
                if result and result[-1]["role"] == "user":
                    prev = result[-1]["content"]
                    if isinstance(prev, list):
                        if content:
                            prev.append({"type": "text", "text": content})
                    else:
                        if content:
                            result[-1]["content"] = (prev + "\n" + content).strip()
                else:
                    result.append({"role": "user", "content": content})
                continue

        while result and result[0]["role"] != "user":
            result.pop(0)

        result = [
            m for m in result
            if m.get("content") not in (None, "", [], [{"type": "text", "text": ""}])
        ]

        return system_parts, result

    @staticmethod
    def _to_fake_message(response) -> "_FakeMessage":
        text_parts: list[str] = []
        tool_calls: list[_FakeToolCall] = []

        for block in response.content:
            if block.type == "text":
                if block.text:
                    text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(_FakeToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input)))

        return _FakeMessage(content="".join(text_parts) or None, tool_calls=tool_calls)

PROVIDERS = {"openai": OpenAIClient, "groq": GroqClient, "gemini": GeminiClient, "claude": ClaudeClient}
_PROVIDER_KEY_ENV = {"openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY",
                     "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


def canonical_provider(name: str) -> str:
    name = (name or "").lower()
    return "claude" if name == "anthropic" else name


def available_providers() -> list[str]:
    """Providers whose API key env var is actually set on this server."""
    return [p for p, env in _PROVIDER_KEY_ENV.items() if os.getenv(env, "").strip()]


@lru_cache(maxsize=None)
def default_provider() -> str:
    """LLM_PROVIDER from .env (read once), falling back to openai if it's unknown."""
    name = canonical_provider(os.getenv("LLM_PROVIDER", "openai"))
    if name not in PROVIDERS:
        logging.warning(f"[STARTUP] Unknown LLM_PROVIDER '{name}'. Valid values: {list(PROVIDERS)}. "
                        "Falling back to 'openai'.")
        return "openai"
    return name


def resolve_provider(requested: str) -> str:
    """The requested provider if its key is set, else the default, else any configured one.
    Never a provider whose key is missing (unless none are configured at all)."""
    avail = available_providers()
    for candidate in (canonical_provider(requested), default_provider()):
        if candidate in avail:
            return candidate
    return avail[0] if avail else "openai"
