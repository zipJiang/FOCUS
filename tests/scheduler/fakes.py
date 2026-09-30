"""The payload and the expander the scheduler tests drive it with, minus a harness.

The scheduler is generic over its payload -- all it asks for is ``forkable`` / ``done``
/ ``reward`` -- and these are the smallest things that satisfy that: a counter that
ends at 3, and an expander that walks one to its terminal. They live here because two
modules run the same scheduler over them (`test_scheduler.py` for the search loop,
`test_critic.py` for the scoring hook), and a test file importing another test file's
fakes makes one of them silently load-bearing for the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from focus.scheduler import Expander


@dataclass(frozen=True)
class Toy:
    """A minimal :class:`Payload`: a counter that is terminal at 3."""

    n: int
    forkable: bool = True

    @property
    def done(self) -> bool:
        return self.n >= 3

    @property
    def reward(self) -> float:
        return float(self.n)


class ChainExpander(Expander[Toy]):
    """Expand a Toy to completion: the ascending chain up to the terminal."""

    async def expand(self, payload: Toy) -> tuple[list[Toy], Mapping[str, object]]:
        chain, p = [], payload
        while not p.done:
            p = Toy(p.n + 1)
            chain.append(p)
        return chain, {}


__all__ = ["ChainExpander", "Toy"]
