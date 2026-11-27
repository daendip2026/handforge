using HandForge.Proto;
using UnityEngine;

// Keeps the generated wire types out of the finger mapping.
public sealed class HandFrame
{
    public const int LandmarkCount = 21;

    private HandFrame(Vector3[] left, Vector3[] right)
    {
        Left = left;
        Right = right;
    }

    /// <summary>The performer's left hand as <see cref="LandmarkCount"/> <see cref="Landmark"/> points in Unity axes, or null if the frame has none.</summary>
    public Vector3[] Left { get; }

    /// <summary>The performer's right hand as <see cref="LandmarkCount"/> <see cref="Landmark"/> points in Unity axes, or null if the frame has none.</summary>
    public Vector3[] Right { get; }

    /// <summary>Converts <paramref name="frame"/>, taking the first hand of each label.</summary>
    /// <exception cref="System.ArgumentOutOfRangeException">A hand has fewer than <see cref="LandmarkCount"/> landmarks.</exception>
    public static HandFrame FromWire(Frame frame)
    {
        Vector3[] left = null;
        Vector3[] right = null;
        foreach (HandPose hand in frame.Hands)
        {
            // MediaPipe can label both hands the same (observed).
            if (hand.Handedness == Handedness.Left && left == null)
                left = ToUnitySpace(hand);
            else if (hand.Handedness == Handedness.Right && right == null)
                right = ToUnitySpace(hand);
        }
        return new HandFrame(left, right);
    }

    private static Vector3[] ToUnitySpace(HandPose hand)
    {
        var points = new Vector3[LandmarkCount];
        for (int i = 0; i < LandmarkCount; i++)
        {
            Landmark p = hand.Landmarks[i];
            // A reflection: landmarks are right-handed with +y down, Unity left-handed with +y up.
            // All three negated so the performer's left, up and forward land on the avatar, which faces +z.
            points[i] = new Vector3(-p.X, -p.Y, -p.Z);
        }
        return points;
    }
}
