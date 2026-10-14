"""Intent is an injected interpretation of the current user request only."""

from __future__ import annotations

from typing import Protocol

from .contracts import IntentHypothesis, Mode, Request


class IntentResolver(Protocol):
    def resolve(self, request: Request) -> tuple[IntentHypothesis, ...]: ...


class ConservativeCurrentIntent:
    """Safe fallback when no interpretation model is configured.

    It can search from the current text/image, but never grants a historical
    entity capability from a substring or a remembered self-description.
    """

    def resolve(self, request: Request) -> tuple[IntentHypothesis, ...]:
        return (IntentHypothesis(Mode.CURRENT),)
