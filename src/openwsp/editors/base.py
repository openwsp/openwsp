from typing import Protocol


class EditorError(RuntimeError):
    pass


class ProviderUnavailable(EditorError):
    """Authentication/quota/provider stage is blocked; do not retry automatically."""


class UncertainRequest(EditorError):
    """Creation may have reached the provider; preserve reservation, do not duplicate."""


class EditorRefused(EditorError):
    """The driver should deliver its protocol refusal message on the next turn."""


class EditorIncomplete(EditorError):
    """A terminal output-limit response was recorded; partial text is not delivered."""

    def __init__(self, reason: str, response_id: str):
        self.reason = reason
        self.response_id = response_id
        super().__init__(f"Editor response incomplete: {reason}; response and usage are recorded")


class Editor(Protocol):
    last_response: dict | None

    def complete(self, messages: str | list[dict]) -> str:
        """Complete this turn using only new input; conversation lives in the adapter."""
        ...

    def new_conversation(self) -> None:
        """Start the next handoff with no previous response and a fresh cache key."""
        ...
