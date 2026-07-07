using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using HandForge.Proto;
using Google.Protobuf;

/* KNOWN GAPS (deferred):
    - Start(): bind failure is unhandled (e.g. port already in use).
    - Class is in the global namespace (should be namespaced).
    - ReceiveLoop(): busy-spins if a socket error persists.
    - OnDestroy(): no null-safety for _udp/_thread if Start() threw before assigning them.
*/
public class FrameReceiver : MonoBehaviour
{
    private const int Port = 9000;
    private const int JoinTimeoutMs = 500; // long enough for the loop to observe the close & exit, short enough not to stall teardown

    private UdpClient _udp;
    private Thread _thread;
    private volatile bool _running;

    // Single slot, not a queue: we only ever want the latest frame,
    // so stale frames are always dropped.
    private Frame _latest;

    private void Start()
    {
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
        // Per-frame allocation (byte[] + Frame) is fine for this proof of concept.
        // If the profiler shows GC hitches, move to a pool of >=2 buffers with ownership
        // handoff — NOT single-instance reuse, which introduces the torn-read race.
        while (_running)
        {
            try
            {
                IPEndPoint remote = new(IPAddress.Any, 0);
                byte[] data = _udp.Receive(ref remote);

                Frame frame = Frame.Parser.ParseFrom(data);

                // Publish through Interlocked (not `_latest = frame`): a plain write isn't
                // guaranteed visible across threads, so the main thread could observe a
                // half-published reference. Interlocked adds the barrier.
                Interlocked.Exchange(ref _latest, frame);
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
        }
    }

    private void Update()
    {
        Frame frame = Interlocked.Exchange(ref _latest, null);
        if (frame == null) return;

        Debug.Log($"FrameIndex: {frame.FrameIndex} , Hands.Count: {frame.Hands.Count}");
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
