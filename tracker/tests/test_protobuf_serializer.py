"""Tests for ProtobufSerializer (ProcessedFrame -> handforge.Frame wire bytes)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from hand_tracker import handtracking_pb2 as pb
from hand_tracker.transport.protobuf import ProtobufSerializer
from hand_tracker.types import (
    LANDMARK_COUNT,
    Handedness,
    ProcessedFrame,
    ProcessedHand,
)

# ---------------------------------------------------------------------------
# Constants & builders
# ---------------------------------------------------------------------------

_BASE_TS: int = 1_700_000_000_000_000  # acquisition timestamp (microseconds)
_SENT_TS: int = _BASE_TS + 5_000  # caller-supplied serialization timestamp


def _world_landmarks(scale: float = 1.0) -> npt.NDArray[np.float32]:
    """Deterministic (21, 3) metric landmarks with recognizable values."""
    data = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    for i in range(LANDMARK_COUNT):
        data[i, 0] = i * 0.01 * scale
        data[i, 1] = i * 0.02 * scale
        data[i, 2] = i * 0.03 * scale
    return data


def _hand(
    handedness: Handedness = Handedness.RIGHT,
    world: npt.NDArray[np.float32] | None = None,
) -> ProcessedHand:
    wl = _world_landmarks() if world is None else world
    # image-space landmarks are deliberately distinct (all -1): they must NOT
    # reach the wire — only world_landmarks do (see /proto/handtracking.proto).
    return ProcessedHand(
        landmarks=np.full((LANDMARK_COUNT, 3), -1.0, dtype=np.float32),
        world_landmarks=wl,
        handedness=handedness,
        confidence=0.9,
    )


def _frame(
    hands: tuple[ProcessedHand, ...],
    frame_index: int = 7,
    timestamp_us: int = _BASE_TS,
) -> ProcessedFrame:
    return ProcessedFrame(
        hands=hands,
        timestamp_us=timestamp_us,
        frame_index=frame_index,
        inference_time_us=8_000,
        fps=30.0,
    )


def _serialize_parse(frame: ProcessedFrame, sent_ts: int = _SENT_TS) -> pb.Frame:
    data = ProtobufSerializer().serialize(frame, sent_ts)
    return pb.Frame.FromString(data)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestFrameRoundTrip:
    """ProcessedFrame -> bytes -> Frame must preserve the wire-relevant fields."""

    def test_frame_metadata_round_trips(self) -> None:
        """Scenario: Nominal frame. Expected: index + both timestamps preserved."""
        frame = _frame((_hand(),), frame_index=42, timestamp_us=_BASE_TS)
        wire = _serialize_parse(frame)
        assert wire.frame_index == 42
        assert wire.acquisition_timestamp_us == _BASE_TS
        assert wire.sent_timestamp_us == _SENT_TS

    def test_only_world_landmarks_reach_the_wire(self) -> None:
        """Scenario: image vs metric differ. Expected: wire carries world_landmarks."""
        world = _world_landmarks(scale=2.0)
        wire = _serialize_parse(_frame((_hand(world=world),)))

        assert len(wire.hands) == 1
        wire_lms = wire.hands[0].landmarks
        assert len(wire_lms) == LANDMARK_COUNT
        for i in range(LANDMARK_COUNT):
            assert wire_lms[i].x == pytest.approx(float(world[i, 0]))
            assert wire_lms[i].y == pytest.approx(float(world[i, 1]))
            assert wire_lms[i].z == pytest.approx(float(world[i, 2]))

    def test_two_hands_preserve_order(self) -> None:
        """Scenario: Dual-hand frame. Expected: order + handedness preserved."""
        wire = _serialize_parse(
            _frame((_hand(Handedness.LEFT), _hand(Handedness.RIGHT)))
        )
        assert len(wire.hands) == 2
        assert wire.hands[0].handedness == pb.HANDEDNESS_LEFT
        assert wire.hands[1].handedness == pb.HANDEDNESS_RIGHT

    def test_empty_frame_serializes_as_no_hands(self) -> None:
        """Scenario: No hand detected. Expected: still a valid Frame with 0 hands."""
        wire = _serialize_parse(_frame((), frame_index=3))
        assert len(wire.hands) == 0
        assert wire.frame_index == 3


# ---------------------------------------------------------------------------
# Handedness boundary mapping
# ---------------------------------------------------------------------------


class TestHandednessMapping:
    """The single domain-enum -> wire-enum boundary, including the BOTH guard."""

    @pytest.mark.parametrize(
        ("domain", "wire"),
        [
            (Handedness.LEFT, pb.HANDEDNESS_LEFT),
            (Handedness.RIGHT, pb.HANDEDNESS_RIGHT),
            (Handedness.UNKNOWN, pb.HANDEDNESS_UNSPECIFIED),
        ],
    )
    def test_domain_enum_maps_to_wire_enum(self, domain: Handedness, wire: int) -> None:
        """Scenario: Each per-hand side. Expected: correct wire enum at the boundary."""
        parsed = _serialize_parse(_frame((_hand(domain),)))
        assert parsed.hands[0].handedness == wire

    def test_both_is_rejected(self) -> None:
        """Scenario: 'BOTH' filter selector leaks to a per-hand result. Expected: raise."""
        with pytest.raises(ValueError, match="BOTH"):
            ProtobufSerializer().serialize(_frame((_hand(Handedness.BOTH),)), _SENT_TS)


# ---------------------------------------------------------------------------
# Clock discipline
# ---------------------------------------------------------------------------


class TestSerializerIsClockFree:
    """The serializer must use the caller's sent timestamp, never read a clock."""

    def test_sent_timestamp_is_caller_supplied(self) -> None:
        """Scenario: Same frame, two sent stamps. Expected: each is echoed verbatim."""
        frame = _frame((_hand(),))
        a = pb.Frame.FromString(ProtobufSerializer().serialize(frame, 111))
        b = pb.Frame.FromString(ProtobufSerializer().serialize(frame, 222))
        assert a.sent_timestamp_us == 111
        assert b.sent_timestamp_us == 222
