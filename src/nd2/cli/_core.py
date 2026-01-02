from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class OperationCancelled(Exception):
    """Exception raised when an operation is cancelled."""


@runtime_checkable
class WorkerSignal(Protocol):
    def emit(self, *args: Any) -> None: ...


class SignalsInterface(Protocol):
    progress: WorkerSignal
    finished: WorkerSignal
    error: WorkerSignal


class DummySignal:
    def emit(self, *args: Any) -> None:
        pass


class DefaultSignals:
    def __init__(self) -> None:
        self.progress = DummySignal()
        self.finished = DummySignal()
        self.error = DummySignal()
