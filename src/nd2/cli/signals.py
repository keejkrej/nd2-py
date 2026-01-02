from __future__ import annotations

from psygnal import Signal, SignalGroup


class ND2Signals(SignalGroup):
    """Signals for ND2 processing events."""

    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)
