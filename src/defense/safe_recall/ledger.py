"""Current-page evidence with product and option version binding."""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import Observation


@dataclass
class EvidenceLedger:
    current: Observation | None = None
    visited: dict[str, list[Observation]] = field(default_factory=dict)

    def add(self, observation: Observation) -> None:
        self.current = observation
        if observation.handle:
            self.visited.setdefault(observation.handle, []).append(observation)

    def current_item(self, handle: str) -> Observation | None:
        item = self.current
        return item if item and item.page == "item" and item.handle == handle else None

    def cited_here(self, observation: Observation, span: str) -> bool:
        return bool(span and span.casefold() in observation.text.casefold())

    def same_binding(self, prior: Observation, current: Observation) -> bool:
        return (
            prior.page == current.page == "item"
            and prior.handle == current.handle
            and prior.version == current.version
            and prior.selected_options == current.selected_options
        )
