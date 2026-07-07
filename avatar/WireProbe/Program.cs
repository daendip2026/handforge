using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

int port = 9000;
using var receiver = new UdpClient(port);
Console.WriteLine($"Listening on port {port}");

while (true)
{
    var result = await receiver.ReceiveAsync();
    var data = result.Buffer;
    var frame = HandForge.Proto.Frame.Parser.ParseFrom(data);

    Console.WriteLine($"Received frame {frame.FrameIndex}");

    foreach (var hand in frame.Hands)
    {
        Console.WriteLine($"Hand {hand.Handedness}");
        foreach (var landmark in hand.Landmarks)
        {
            Console.WriteLine($"Landmark {landmark.X} {landmark.Y} {landmark.Z}");
        }
    }
}
