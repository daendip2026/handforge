"""
UDP data-plane transport (send-only).

One datagram per 'Frame'. UDP preserves message boundaries (one 'recvfrom' == one message),
so no length-prefix framing is needed here — unlike the TCP control plane.

Non-blocking by design: 'sendto' never waits on a slow consumer,
and a full kernel send buffer surfaces as 'BlockingIOError' rather than a stall.
We drop that datagram; UDP offers no delivery guarantee and the next frame supersedes
a lost one anyway.
"""

from __future__ import annotations

import logging
import socket
from types import TracebackType

from hand_tracker.transport.base import ITransport

log = logging.getLogger(__name__)


class UdpTransport(ITransport):
    """Send-only UDP transport to a fixed (host, port) destination."""

    def __init__(self, host: str, port: int) -> None:
        self._addr: tuple[str, int] = (host, port)
        self._sock: socket.socket | None = None

    def __enter__(self) -> UdpTransport:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Send-only client: no bind(). The OS assigns an ephemeral source port.
        # Non-blocking so a full send buffer raises instead of stalling.
        sock.setblocking(False)
        self._sock = sock
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("UdpTransport.send() called before __enter__().")
        try:
            self._sock.sendto(data, self._addr)
        except BlockingIOError:
            # Kernel send buffer full under load: drop this datagram.
            log.warning("udp send buffer full; dropping frame")
