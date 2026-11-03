from typing import Protocol


class EditorError(RuntimeError):
    pass


class ProviderUnavailable(EditorError):
    """Authentication/quota/provider stage is blocked; do not retry automatically."""


class UncertainRequest(EditorError):
    """Creation may have reached the provider; preserve reservation, do not duplicate."""


class EditorRefused(EditorError):
    """The driver should deliver its protocol refusal message on the next turn."""


class Editor(Protocol):
    last_response: dict | None

    def complete(self, messages: str | list[dict]) -> str:
        """Complete this turn using only new input; conversation lives in the adapter."""
        ...

    def new_conversation(self) -> None:
        """Start the next handoff with no previous response and a fresh cache key."""
        ...
