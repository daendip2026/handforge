"""
Protobuf serializer for the HandForge data plane.

Maps the tracker's canonical output ('ProcessedFrame') to the wire 'Frame'.

Wire decisions embodied here (see /proto/handtracking.proto):
- Only 'world_landmarks' (hand-centric metric) go on the wire; image-space
  landmarks and per-hand confidence are intentionally omitted.
- Domain 'Handedness' is mapped to the 3-value wire enum at this single
  boundary; the filter-intent value 'BOTH' must never reach a per-hand result
  and raises if it does.
"""

from __future__ import annotations

from hand_tracker import handtracking_pb2 as pb
from hand_tracker.transport.base import ISerializer
from hand_tracker.types import Handedness, ProcessedFrame

# The one boundary mapping: domain enum -> wire enum.
#   LEFT / RIGHT      -> same
#   UNKNOWN (per-hand "side undetermined") -> UNSPECIFIED
#   BOTH (filter selector) -> absent on purpose; .get() returns None -> raise.
_HANDEDNESS_TO_WIRE: dict[Handedness, pb.Handedness] = {
    Handedness.LEFT: pb.HANDEDNESS_LEFT,
    Handedness.RIGHT: pb.HANDEDNESS_RIGHT,
    Handedness.UNKNOWN: pb.HANDEDNESS_UNSPECIFIED,
}


class ProtobufSerializer(ISerializer):
    """Serialize a  'ProcessedFrame' to 'handforge.Frame' bytes."""

    def serialize(self, frame: ProcessedFrame, sent_timestamp_us: int) -> bytes:
        msg = pb.Frame(
            frame_index=frame.frame_index,
            acquisition_timestamp_us=frame.timestamp_us,
            sent_timestamp_us=sent_timestamp_us,
        )

        for hand in frame.hands:
            wire_side = _HANDEDNESS_TO_WIRE.get(hand.handedness)
            if wire_side is None:
                raise ValueError(
                    f"Handedness {hand.handedness!r} cannot be serialized as a "
                    "per-hand result. 'BOTH' is a filter selector and must never "
                    "reach the wire mapping (programming error upstream)."
                )

            hp = msg.hands.add()
            hp.handedness = wire_side
            # world_landmarks: np.float32 array of shape (21, 3) -> 21 Landmarks.
            # Per-frame cost is ~42 add() calls; negligible next to inference
            # (tens of ms), so this is intentionally not micro-optimized.
            for row in hand.world_landmarks:
                hp.landmarks.add(x=float(row[0]), y=float(row[1]), z=float(row[2]))

        return msg.SerializeToString()
