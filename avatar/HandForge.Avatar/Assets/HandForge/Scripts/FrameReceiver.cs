  using System;
  using System.Net;
  using System.Net.Sockets;
  using System.Threading;
  using UnityEngine;
  using HandForge.Proto;

  public class FrameReceiver : MonoBehaviour
  {
      private const int Port = 9000;             

      private UdpClient _udp;                      
      private Thread _thread;                      
      private volatile bool _running;             
      private Frame _latest;                      

      private void Start()
      {
          _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, Port));
          _running = true;
          _thread = new Thread(ReceiveLoop) {
            IsBackground = true
          };
          _thread.Start();
      }

      private void ReceiveLoop()
      {
          while(_running) {
            try{
              IPEndPoint remote = new(IPAddress.Any, 0);
              byte[] data = _udp.Receive(ref remote);

              Frame frame = Frame.Parser.ParseFrom(data);
              Interlocked.Exchange(ref _latest, frame);
            }
            catch (System.Exception ex) when (ex is SocketException or ObjectDisposedException){
              if (_running) 
                Debug.LogError("FrameReceiver : SocketException or ObjectDisposedException in ReceiveLoop: " + ex.Message);
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
          _udp.Close();
          _thread.Join(500);
      }
  }