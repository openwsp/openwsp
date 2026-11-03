"""Immutable model choices and an extensible registry of editor transports.

Profiles select a transport explicitly. They never infer prices, credentials,
provider entitlement, or permission to fall back to another billing account.
"""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..budget import atomic_json
from .base import Editor, EditorError


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be nonempty single-line text")


@dataclass(frozen=True, kw_only=True)
class EditorProfile:
    name: str
    transport: str
    model: str

    def __post_init__(self):
        for name in ("name", "transport", "model"):
            _text(getattr(self, name), name)

    def record(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


@dataclass(frozen=True, kw_only=True)
class CodexProfile(EditorProfile):
    cli_version: str
    catalog_path: Path
    catalog_sha256: str
    effort: str
    verbosity: str | None
    instructions: str
    context_window: int
    auto_compact_token_limit: int

    def __post_init__(self):
        super().__post_init__()
        if self.transport != "codex_cli_chatgpt":
            raise ValueError("CodexProfile requires the codex_cli_chatgpt transport")
        _text(self.cli_version, "cli_version")
        _text(self.effort, "effort")
        if self.verbosity not in (None, "low", "medium", "high"):
            raise ValueError("Unsupported Codex verbosity")
        if not isinstance(self.catalog_sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.catalog_sha256):
            raise ValueError("catalog_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "catalog_path", Path(self.catalog_path))
        if not isinstance(self.instructions, str):
            raise TypeError("instructions must be text")
        _text(self.instructions.removesuffix("\n"), "instructions")
        if any(
            type(value) is not int or value <= 0
            for value in (self.context_window, self.auto_compact_token_limit)
        ):
            raise ValueError("Context and compaction limits must be positive integers")
        if self.auto_compact_token_limit < self.context_window:
            raise ValueError("Compaction limit must cover the explicit context window")


@dataclass(frozen=True, kw_only=True)
class ThinkingConfig:
    type: str
    budget_tokens: int | None = None

    def __post_init__(self):
        if self.type not in ("disabled", "adaptive", "enabled"):
            raise ValueError("Unsupported explicit thinking configuration")
        if self.type == "enabled":
            if type(self.budget_tokens) is not int or self.budget_tokens < 1024:
                raise ValueError("Enabled thinking requires at least 1024 budget tokens")
        elif self.budget_tokens is not None:
            raise ValueError("A thinking budget is valid only for enabled thinking")

    def request(self):
        return {
            "type": self.type,
            **({"budget_tokens": self.budget_tokens} if self.budget_tokens is not None else {}),
        }


@dataclass(frozen=True, kw_only=True)
class AnthropicMessagesProfile(EditorProfile):
    thinking: ThinkingConfig | None = None
    output_effort: str | None = None
    system: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.transport != "anthropic_messages":
            raise ValueError("AnthropicMessagesProfile requires the anthropic_messages transport")
        if self.thinking is not None and not isinstance(self.thinking, ThinkingConfig):
            raise TypeError("thinking must be an immutable ThinkingConfig")
        if self.output_effort not in (None, "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported Anthropic output effort")
        if self.system is not None and not isinstance(self.system, str):
            raise TypeError("system must be text")


@dataclass(frozen=True, kw_only=True)
class AstraResponsesProfile(EditorProfile):
    effort: str = "max"
    verbosity: str | None = None
    max_output_tokens: int = 32768

    def __post_init__(self):
        super().__post_init__()
        if self.transport != "astra_responses" or self.model != "gpt-6-astra":
            raise ValueError("The existing Responses adapter verifies Astra pricing only")
        if self.effort not in ("max", "xhigh", "high", "medium", "low"):
            raise ValueError("Unsupported Astra effort")
        if self.verbosity not in (None, "low", "medium", "high"):
            raise ValueError("Unsupported Astra verbosity")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 128000:
            raise ValueError("Invalid Astra output token ceiling")


ASTRA_CODEX_PROFILE = CodexProfile(
    name="astra-codex-0.163.0",
    transport="codex_cli_chatgpt",
    model="gpt-6-astra",
    cli_version="0.163.0",
    catalog_path=Path(__file__).parent / "data" / "codex-models-0.163.0.json",
    catalog_sha256="05723cac0f7da85c31f710f602bb0c47a5588d8bdf68aad24e9dd6ac951c7705",
    effort="max",
    verbosity="low",
    instructions="Answer the user's request.\n",
    context_window=872000,
    auto_compact_token_limit=1000000,
)
ASTRA_RESPONSES_PROFILE = AstraResponsesProfile(
    name="astra-responses",
    transport="astra_responses",
    model="gpt-6-astra",
)
FABLE_5_1_PROFILE = AnthropicMessagesProfile(
    name="fable-5.1-native",
    transport="anthropic_messages",
    model="claude-fable-5-1",
    system="Answer the user's request.",
)
# Reference identity only. No OpenRouter transport or native-price inference is installed.
FABLE_OPENROUTER_MODEL_ID = "anthropic/claude-fable-5.1"


class EditorRegistry:
    """Register transport builders at application startup; duplicate IDs are refused."""

    def __init__(self):
        self._builders: dict[str, Callable[..., Editor]] = {}

    def register(self, transport: str, builder: Callable[..., Editor]):
        _text(transport, "transport")
        if not callable(builder):
            raise TypeError("Transport builder must be callable")
        if transport in self._builders:
            raise ValueError(f"Editor transport is already registered: {transport}")
        self._builders[transport] = builder

    def create(self, profile: EditorProfile, **options) -> Editor:
        if not isinstance(profile, EditorProfile):
            raise TypeError("An explicit immutable EditorProfile is required")
        try:
            builder = self._builders[profile.transport]
        except KeyError:
            raise ValueError(f"No editor transport is registered for {profile.transport}") from None
        return builder(profile, **options)


def _codex(profile, **options):
    from .codex import CodexEditor

    if not isinstance(profile, CodexProfile):
        raise TypeError("The Codex transport requires CodexProfile")
    return CodexEditor(profile=profile, **options)


def _anthropic(profile, **options):
    from .anthropic import AnthropicEditor

    if not isinstance(profile, AnthropicMessagesProfile):
        raise TypeError("The Anthropic transport requires AnthropicMessagesProfile")
    return AnthropicEditor(
        model=profile.model,
        thinking=profile.thinking.request() if profile.thinking else None,
        output_effort=profile.output_effort,
        system=profile.system,
        **options,
    )


def _astra_responses(profile, **options):
    from .openai import AstraEditor

    if not isinstance(profile, AstraResponsesProfile):
        raise TypeError("The Astra Responses transport requires AstraResponsesProfile")
    editor = AstraEditor(
        model=profile.model,
        effort=profile.effort,
        verbosity=profile.verbosity,
        max_output_tokens=profile.max_output_tokens,
        **options,
    )
    binding_path = editor.root / "responses_profile.json"
    expected = {"schema_version": 1, "profile": profile.record()}
    with (editor.root / "editor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EditorError("Responses editor is busy; profile binding cannot be checked") from None
        if binding_path.exists() or binding_path.is_symlink():
            try:
                if binding_path.is_symlink() or json.loads(binding_path.read_text()) != expected:
                    raise ValueError("Profile differs")
            except (OSError, ValueError):
                raise EditorError(
                    "Responses profile binding differs or is unreadable; use a new state directory"
                ) from None
        else:
            if (
                editor.path.exists()
                or any(editor.root.glob("request_*.json"))
                or any(editor.root.glob("receipt_*.json"))
            ):
                raise EditorError(
                    "Existing API state has no profile binding; continue with AstraEditor or use a new state directory"
                )
            atomic_json(binding_path, expected)
        editor.commit()
    return editor


def default_registry():
    registry = EditorRegistry()
    registry.register("codex_cli_chatgpt", _codex)
    registry.register("anthropic_messages", _anthropic)
    registry.register("astra_responses", _astra_responses)
    return registry


EDITOR_REGISTRY = default_registry()


def create_editor(profile: EditorProfile, **options) -> Editor:
    return EDITOR_REGISTRY.create(profile, **options)


def register_editor_transport(transport: str, builder: Callable[..., Editor]):
    EDITOR_REGISTRY.register(transport, builder)
