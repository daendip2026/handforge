using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using UnityEngine.InputSystem;
using HandForge.Proto;
using Google.Protobuf;

/* KNOWN GAPS (deferred):
    - Start(): bind failure is unhandled (e.g. port already in use).
    - Class is in the global namespace (should be namespaced).
    - ReceiveLoop(): busy-spins if a socket error persists.
    - OnDestroy(): no null-safety for _udp/_thread if Start() returned early or threw before assigning them.
*/
public class FrameReceiver : MonoBehaviour
{
    private const int Port = 9000;
    private const int JoinTimeoutMs = 500; // long enough for the loop to observe the close & exit, short enough not to stall teardown

    // One key per hand, so the hand pressing it is not recorded: L is under the right hand, R the left.
    // Input System keys are physical positions named after the US layout (Input System docs).
    private const Key RecordLeftHandKey = Key.L;
    private const Key RecordRightHandKey = Key.R;

    private FingerMapping _leftFingers;
    private FingerMapping _rightFingers;
    private UdpClient _udp;
    private Thread _thread;
    private volatile bool _running;

    // Single slot, not a queue: we only ever want the latest frame,
    // so stale frames are always dropped.
    private HandFrame _latest;

    // Kept across Updates: a key pressed in an Update with no new frame must still record.
    private HandFrame _hands;

    private void Start()
    {
        var humanoidRig = GetComponent<Animator>();
        if (!FingerMapping.TryCreate(humanoidRig, FingerMapping.LeftHand, out _leftFingers, out HumanBodyBones missing)
            || !FingerMapping.TryCreate(humanoidRig, FingerMapping.RightHand, out _rightFingers, out missing))
        {
            enabled = false;
            Debug.LogError(missing + " is not found");
            return;
        }

        _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, Port));
        _running = true;
        _thread = new Thread(ReceiveLoop)
        {
            IsBackground = true
        };
        _thread.Start();
    }

    private void ReceiveLoop()
    {
        // Per-datagram allocation (byte[], Frame, HandFrame) is accepted.
        // If the profiler shows GC hitches, move to a pool of >=2 buffers with ownership
        // handoff — NOT single-instance reuse, which introduces the torn-read race.
        bool receivedFirstFrame = false;
        while (_running)
        {
            try
            {
                IPEndPoint remote = new(IPAddress.Any, 0);
                byte[] data = _udp.Receive(ref remote);

                Frame frame = Frame.Parser.ParseFrom(data);
                if (!receivedFirstFrame)
                {
                    receivedFirstFrame = true;
                    Debug.Log("FrameReceiver: received the first Frame on UDP port " + Port);
                }

                HandFrame hands = HandFrame.FromWire(frame);

                // Not a plain store: this code relies only on ECMA-335, which orders a plain store only within its thread (I.12.6.4).
                Interlocked.Exchange(ref _latest, hands);
            }
            catch (Exception ex) when (ex is SocketException or ObjectDisposedException)
            {
                if (_running)
                    Debug.LogError("FrameReceiver : SocketException or ObjectDisposedException in ReceiveLoop: " + ex.Message);
            }
            catch (InvalidProtocolBufferException ex)
            {
                Debug.LogWarning("Dropped a malformed datagram (protobuf parse failed): " + ex.Message);
            }
            catch (Exception ex)
            {
                Debug.LogError("Dropped a datagram after an unexpected exception: " + ex);
            }
        }
    }

    private void Update()
    {
        HandFrame received = Interlocked.Exchange(ref _latest, null);
        if (received != null) _hands = received;
        if (_hands == null) return;

        Keyboard keyboard = Keyboard.current;
        if (keyboard != null)
        {
            if (keyboard[RecordLeftHandKey].wasPressedThisFrame && _hands.Left != null)
                _leftFingers.RecordOpenHand(_hands.Left);
            if (keyboard[RecordRightHandKey].wasPressedThisFrame && _hands.Right != null)
                _rightFingers.RecordOpenHand(_hands.Right);
        }

        if (_hands.Left != null) _leftFingers.Apply(_hands.Left);
        if (_hands.Right != null) _rightFingers.Apply(_hands.Right);
    }

    private void OnDestroy()
    {
        _running = false;

        // If the thread is blocked in Receive(), closing the socket unblocks it (throws),
        // so it can see _running == false and exit. A flag alone can't wake a blocked
        // Receive() — that's why we Close() before Join().
        _udp.Close();
        _thread.Join(JoinTimeoutMs);
    }
}
