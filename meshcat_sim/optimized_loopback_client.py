from zeroconf import Zeroconf, ServiceBrowser
import socket
import zmq
import time
import msgpack
import numpy as np

# Zeroconf service type identifier
SERVICE_TYPE = "_loopback._tcp.local."


class ZeroconfListener:
    """
    Listener class for handling Zeroconf service events.
    Captures the IP address and port when the target service is discovered.
    """

    def __init__(self):
        self.server_ip = None
        self.port = None

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            self.server_ip = socket.inet_ntoa(info.addresses[0])
            self.port = info.port
            print(f"🚀 Zeroconf discovered: {self.server_ip}:{self.port}")

    def remove_service(self, zeroconf, type, name):
        pass

    def update_service(self, zeroconf, type, name):
        pass


def discover_with_zeroconf(timeout=3):
    zeroconf = Zeroconf()
    listener = ZeroconfListener()
    browser = ServiceBrowser(zeroconf, SERVICE_TYPE, listener)
    start = time.time()
    while not listener.server_ip and (time.time() - start) < timeout:
        time.sleep(0.1)
    zeroconf.close()
    return listener.server_ip, listener.port


def discover_with_udp(timeout=3):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    sock.sendto(b"where_is_loopback", ("<broadcast>", 50000))
    try:
        data, addr = sock.recvfrom(1024)
        if data == b"loopback_here":
            print(f"📱 UDP discovered: {addr[0]}:5555")
            return addr[0], 5555
    except socket.timeout:
        return None, None


def main():
    ip, port = discover_with_zeroconf()
    if not ip:
        print("⚠️ Zeroconf failed. Trying UDP fallback...")
        ip, port = discover_with_udp()

    if not ip:
        print("❌ Failed to find loopback server.")
        return

    print(f"🔗 Connecting to loopback server at {ip}:{port}")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    # Disable Nagle’s algorithm for lower latency
    socket.setsockopt(zmq.TCP_NODELAY, 1)
    socket.setsockopt(zmq.LINGER, 0)       # Don’t wait on close
    socket.connect(f"tcp://{ip}:{port}")

    # Generate and serialize a float32 NumPy array using msgpack
    array = np.random.rand(4, 4).astype(np.float32)
    metadata = {
        "shape": array.shape,
        "dtype": str(array.dtype)
    }
    packed = msgpack.packb({"meta": metadata, "data": array.tolist()})
    socket.send(packed)

    # Receive and unpack the echoed message
    response = msgpack.unpackb(socket.recv(), raw=False)
    result_array = np.array(response["data"], dtype=response["meta"]["dtype"])

    print("📤 Sent:\n", array)
    print("📥 Received back:\n", result_array)


if __name__ == "__main__":
    main()
