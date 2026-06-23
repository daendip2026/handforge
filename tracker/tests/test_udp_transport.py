"""Tests for UdpTransport (send-only UDP data plane) over a loopback socket."""

from __future__ import annotations

import socket
from collections.abc import Iterator

import numpy as np
import pytest

from hand_tracker import handtracking_pb2 as pb
from hand_tracker.transport.protobuf import ProtobufSerializer
from hand_tracker.transport.udp import UdpTransport
from hand_tracker.types import (
    LANDMARK_COUNT,
    Handedness,
    ProcessedFrame,
    ProcessedHand,
)

_HOST = "127.0.0.1"


@pytest.fixture
def udp_receiver() -> Iterator[tuple[socket.socket, int]]:
    """A bound loopback UDP socket; yields (socket, ephemeral_port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((_HOST, 0))
    sock.settimeout(1.0)
    port: int = sock.getsockname()[1]
    try:
        yield sock, port
    finally:
        sock.close()


class TestUdpTransport:
    """Lifecycle + delivery of the send-only UDP transport."""

    def test_send_delivers_payload(
        self, udp_receiver: tuple[socket.socket, int]
    ) -> None:
        """Scenario: One datagram. Expected: receiver gets identical bytes."""
        recv_sock, port = udp_receiver
        payload = b"handforge-frame-bytes"
        with UdpTransport(_HOST, port) as transport:
            transport.send(payload)
        data, _ = recv_sock.recvfrom(4096)
        assert data == payload

    def test_send_before_enter_raises(self) -> None:
        """Scenario: send() before resource acquisition. Expected: RuntimeError."""
        transport = UdpTransport(_HOST, 9000)
        with pytest.raises(RuntimeError, match="__enter__"):
            transport.send(b"x")

    def test_send_after_exit_raises(
        self, udp_receiver: tuple[socket.socket, int]
    ) -> None:
        """Scenario: send() after context closes. Expected: socket released, raise."""
        _recv, port = udp_receiver
        transport = UdpTransport(_HOST, port)
        with transport:
            transport.send(b"ok")
        with pytest.raises(RuntimeError, match="__enter__"):
            transport.send(b"too late")

    def test_multiple_datagrams_delivered(
        self, udp_receiver: tuple[socket.socket, int]
    ) -> None:
        """Scenario: Several frames in a row. Expected: all delivered over loopback."""
        recv_sock, port = udp_receiver
        payloads = [f"frame-{i}".encode() for i in range(5)]
        with UdpTransport(_HOST, port) as transport:
            for p in payloads:
                transport.send(p)
        received: list[bytes] = []
        for _ in payloads:
            data, _ = recv_sock.recvfrom(4096)
            received.append(data)
        assert set(received) == set(payloads)


class TestDataPlaneRoundTrip:
    """Serializer + transport together = the data plane, verified without the app."""

    def test_processed_frame_reaches_wire_as_frame(
        self, udp_receiver: tuple[socket.socket, int]
    ) -> None:
        """Scenario: Full data-plane path. Expected: ProcessedFrame decodes on the wire."""
        recv_sock, port = udp_receiver
        world = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
        world[0] = (0.1, 0.2, 0.3)
        frame = ProcessedFrame(
            hands=(
                ProcessedHand(
                    landmarks=world,
                    world_landmarks=world,
                    handedness=Handedness.RIGHT,
                    confidence=0.9,
                ),
            ),
            timestamp_us=111,
            frame_index=300,
            inference_time_us=0,
            fps=30.0,
        )

        data = ProtobufSerializer().serialize(frame, sent_timestamp_us=222)
        with UdpTransport(_HOST, port) as transport:
            transport.send(data)

        received, _ = recv_sock.recvfrom(65535)
        wire = pb.Frame.FromString(received)
        assert wire.frame_index == 300
        assert wire.sent_timestamp_us == 222
        assert len(wire.hands) == 1
        assert wire.hands[0].handedness == pb.HANDEDNESS_RIGHT
        assert len(wire.hands[0].landmarks) == LANDMARK_COUNT
        assert wire.hands[0].landmarks[0].x == pytest.approx(0.1)
