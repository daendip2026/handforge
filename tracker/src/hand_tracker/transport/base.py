"""
Transport-layer contracts for HandForge.

These two interfaces sit at the boundary between the tracker domain and the network:

- ISerializer is the single place a domain object (ProcessedFrame)
  becomes wire bytes. It is the only transport-side type that imports a domain type.
- ITransport carries opaque bytes and never imports a domain type.

That asymmetry is deliberate: it lets the output backend change (e.g. a future
plan distributed deployment) by touching ``transport/`` alone, with the
tracker domain untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

    from hand_tracker.types import ProcessedFrame


class ISerializer(ABC):
    """Domain -> wire. The one place 'ProcessedFrame' becomes bytes."""

    @abstractmethod
    def serialize(self, frame: ProcessedFrame, sent_timestamp_us: int) -> bytes:
        """
        Serialize one 'ProcessedFrame' into a single data-plane datagram.

        'sent_timestamp_us' is supplied by the caller (not read from a clock
        here) so it shares the same monotonic clock basis as the frame's
        acquisition timestamp; otherwise the acquisition->sent delta is not a
        valid latency. Keeping the serializer clock-free also keeps it pure and
        unit-testable.
        """
        ...


class ITransport(ABC):
    """Carries opaque bytes. Never imports or inspects a domain type."""

    @abstractmethod
    def send(self, data: bytes) -> None:
        """
        Send one message.

        May raise on a transport-level failure. Callers on the inference hot
        path must treat a failed send as a dropped frame (log + continue), never
        letting it stall the loop. The data-plane (UDP) implementation also
        absorbs transient buffer-full conditions internally.
        """
        ...

    @abstractmethod
    def __enter__(self) -> ITransport:
        """Acquire transport resources (TCP connect / UDP socket creation)."""
        ...

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release transport resources deterministically."""
        ...
