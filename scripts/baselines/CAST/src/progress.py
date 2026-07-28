from __future__ import annotations

import sys
from typing import Any, Iterable, Iterator, TextIO, TypeVar

from tqdm import tqdm


T = TypeVar("T")


class StageProgress:
    """Terminal step progress shared by all four stages, written to stderr to preserve stdout JSON."""

    def __init__(
        self,
        stage: str,
        total: int,
        *,
        file: TextIO | None = None,
    ) -> None:
        self._closed = True
        if total <= 0:
            raise ValueError("progress total must be greater than 0.")
        self.total = int(total)
        self._active_label = "Initializing"
        self._bar = tqdm(
            total=self.total,
            desc=f"CAST {stage}",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            file=file or sys.stderr,
        )
        self._closed = False
        self.begin(self._active_label)

    def begin(self, label: str) -> None:
        self._active_label = str(label)
        self._bar.set_postfix_str(self._active_label, refresh=True)

    def advance(self, steps: int = 1) -> None:
        if self._closed:
            return
        self._bar.update(min(int(steps), self.total - self._bar.n))

    def finish(self) -> None:
        if self._closed:
            return
        if self._bar.n < self.total:
            self._bar.update(self.total - self._bar.n)
        self._bar.set_postfix_str("Complete", refresh=True)
        self._bar.close()
        self._closed = True

    def close(self) -> None:
        if not self._closed:
            self._bar.close()
            self._closed = True

    def __del__(self) -> None:
        self.close()


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = False,
) -> Iterator[T]:
    """Provide a unified dynamic-width tqdm for long loops."""
    return iter(
        tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            dynamic_ncols=True,
            leave=leave,
        )
    )
