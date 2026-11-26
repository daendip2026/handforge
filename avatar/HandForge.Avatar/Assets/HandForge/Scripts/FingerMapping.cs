using UnityEngine;

// Directions are relative to the hand bone, so the fingers follow it.
// Landmarks give no twist, so the shortest arc keeps the rest twist.
public sealed class FingerMapping
{
    // Unity's thumb bones are VRM's thumbMetacarpal/Proximal/Distal (UniVRM): landmarks 1-2, 2-3, 3-4.
    public static readonly HumanBodyBones[] LeftHand =
    {
        HumanBodyBones.LeftHand,
        HumanBodyBones.LeftThumbProximal,
        HumanBodyBones.LeftThumbIntermediate,
        HumanBodyBones.LeftThumbDistal,
        HumanBodyBones.LeftIndexProximal,
        HumanBodyBones.LeftIndexIntermediate,
        HumanBodyBones.LeftIndexDistal,
        HumanBodyBones.LeftMiddleProximal,
        HumanBodyBones.LeftMiddleIntermediate,
        HumanBodyBones.LeftMiddleDistal,
        HumanBodyBones.LeftRingProximal,
        HumanBodyBones.LeftRingIntermediate,
        HumanBodyBones.LeftRingDistal,
        HumanBodyBones.LeftLittleProximal,
        HumanBodyBones.LeftLittleIntermediate,
        HumanBodyBones.LeftLittleDistal,
    };

    public static readonly HumanBodyBones[] RightHand =
    {
        HumanBodyBones.RightHand,
        HumanBodyBones.RightThumbProximal,
        HumanBodyBones.RightThumbIntermediate,
        HumanBodyBones.RightThumbDistal,
        HumanBodyBones.RightIndexProximal,
        HumanBodyBones.RightIndexIntermediate,
        HumanBodyBones.RightIndexDistal,
        HumanBodyBones.RightMiddleProximal,
        HumanBodyBones.RightMiddleIntermediate,
        HumanBodyBones.RightMiddleDistal,
        HumanBodyBones.RightRingProximal,
        HumanBodyBones.RightRingIntermediate,
        HumanBodyBones.RightRingDistal,
        HumanBodyBones.RightLittleProximal,
        HumanBodyBones.RightLittleIntermediate,
        HumanBodyBones.RightLittleDistal,
    };

    private const int FingerCount = 5;
    private const int JointsPerFinger = 3;
    private const int IndexProximal = 3;
    private const int LittleProximal = 12;

    // MediaPipe landmark of each finger's first joint (Hand Landmarker docs).
    private static readonly int[] FirstLandmarkOfFinger = { 1, 5, 9, 13, 17 };

    private readonly Transform[] _bones;
    private readonly Quaternion[] _restLocalRotations;
    private readonly Vector3[] _restDirections;
    private readonly Quaternion _palmInHand;

    private FingerMapping(Transform[] bones)
    {
        _bones = bones;
        _restLocalRotations = new Quaternion[bones.Length];
        _restDirections = new Vector3[bones.Length];
        for (int k = 0; k < bones.Length; k++)
        {
            bool distal = k % JointsPerFinger == JointsPerFinger - 1;
            // ...Distal bones have no child; each is assumed to point the way its parent points at it,
            // as the other finger bones of this avatar do (measured).
            Vector3 direction = distal
                ? bones[k].localPosition.normalized
                : bones[k + 1].localPosition.normalized;
            _restLocalRotations[k] = bones[k].localRotation;
            _restDirections[k] = bones[k].localRotation * direction;
        }
        _palmInHand = Palm(
            Vector3.zero,
            bones[IndexProximal].localPosition,
            bones[LittleProximal].localPosition);
    }

    /// <summary>Finds the bones of <paramref name="hand"/> (<see cref="LeftHand"/> or <see cref="RightHand"/>) and takes their current rotations as the rest pose; on false, <paramref name="missing"/> is the first bone not found.</summary>
    public static bool TryCreate(
        Animator animator,
        HumanBodyBones[] hand,
        out FingerMapping mapping,
        out HumanBodyBones missing)
    {
        mapping = null;
        missing = hand[0];
        if (animator.GetBoneTransform(hand[0]) == null)
            return false;

        var bones = new Transform[FingerCount * JointsPerFinger];
        for (int k = 0; k < bones.Length; k++)
        {
            bones[k] = animator.GetBoneTransform(hand[k + 1]);
            if (bones[k] == null)
            {
                missing = hand[k + 1];
                return false;
            }
        }
        mapping = new FingerMapping(bones);
        return true;
    }

    /// <summary>Rotates the finger bones to follow <paramref name="points"/>, one hand from <see cref="HandFrame"/>; the hand bone is not changed.</summary>
    public void Apply(Vector3[] points)
    {
        Quaternion toHand = ToHand(points);
        for (int f = 0; f < FingerCount; f++)
        {
            Quaternion parent = Quaternion.identity;
            for (int j = 0; j < JointsPerFinger; j++)
            {
                int k = f * JointsPerFinger + j;
                Vector3 target = Direction(points, toHand, parent, f, j);
                Quaternion local = ShortestArc(_restDirections[k], target)
                    * _restLocalRotations[k];
                _bones[k].localRotation = local;
                parent *= local;
            }
        }
    }

    private Quaternion ToHand(Vector3[] points)
    {
        return _palmInHand * Quaternion.Inverse(Palm(points[0], points[5], points[17]));
    }

    // In the bone's parent space; `parent` is relative to the hand bone.
    private static Vector3 Direction(
        Vector3[] points, Quaternion toHand, Quaternion parent, int finger, int joint)
    {
        int a = FirstLandmarkOfFinger[finger] + joint;
        return Quaternion.Inverse(parent) * (toHand * (points[a + 1] - points[a]));
    }

    private static Quaternion Palm(Vector3 wrist, Vector3 indexRoot, Vector3 littleRoot)
    {
        Vector3 x = indexRoot - wrist;
        Vector3 z = Vector3.Cross(x, littleRoot - wrist);
        return Quaternion.LookRotation(z, Vector3.Cross(z, x));
    }

    private static Quaternion ShortestArc(Vector3 from, Vector3 to)
    {
        Vector3 a = from.normalized;
        Vector3 b = to.normalized;
        float w = 1f + Vector3.Dot(a, b);
        // A 180-degree bend from rest is treated as a tracking error; any perpendicular axis is used.
        if (w < 1e-6f)
        {
            Vector3 other = Mathf.Abs(a.x) < 0.9f ? Vector3.right : Vector3.up;
            Vector3 axis = Vector3.Cross(a, other).normalized;
            return new Quaternion(axis.x, axis.y, axis.z, 0f);
        }
        Vector3 c = Vector3.Cross(a, b);
        return new Quaternion(c.x, c.y, c.z, w).normalized;
    }
}
