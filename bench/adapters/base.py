from __future__ import annotations

from typing import Protocol

from bench.schema import Case


class Adapter(Protocol):
    """What the runner needs from a model: load it, answer a case, optionally free it."""

    name: str

    def warmup(self) -> None: ...

    def decide(self, case: Case) -> str:
        """Return the model's raw answer for the case."""
