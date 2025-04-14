from zeroconf import Zeroconf, ServiceBrowser
import socket
import zmq
import time
import pickle
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
        """
        Called when a new service is found.
        Extracts the IP and port of the discovered server.
        """
        info = zeroconf.get_service_info(type, name)
        if info:
            self.server_ip = socket.inet_ntoa(info.addresses[0])
            self.port = info.port
            print(f"🚀 Zeroconf discovered: {self.server_ip}:{self.port}")

    def remove_service(self, zeroconf, type, name):
        """
        Called when a service is removed from the network.
        Not used in this implementation.
        """
        pass

    def update_service(self, zeroconf, type, name):
        """
        Called when a service is updated.
        Not used here, but required by the API.
        """
        pass


def discover_with_zeroconf(timeout=3):
    """
    Discovers a service on the local network using Zeroconf/mDNS.

    Args:
        timeout (int): Number of seconds to wait for discovery.

    Returns:
        tuple: (IP address, port) of the discovered server, or (None, None).
    """
    zeroconf = Zeroconf()
    listener = ZeroconfListener()
    browser = ServiceBrowser(zeroconf, SERVICE_TYPE, listener)
    start = time.time()
    while not listener.server_ip and (time.time() - start) < timeout:
        time.sleep(0.1)
    zeroconf.close()
    return listener.server_ip, listener.port


def discover_with_udp(timeout=3):
    """
    Broadcasts a UDP message on the local network to discover the server
    in case Zeroconf is unavailable or fails.

    Args:
        timeout (int): Number of seconds to wait for a response.

    Returns:
        tuple: (IP address, port) of the server, or (None, None).
    """
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
    """
    Main routine that discovers the server, connects to it over ZeroMQ,
    sends a NumPy array, and prints the echoed result.
    """
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
    socket.connect(f"tcp://{ip}:{port}")

    # Create a test NumPy array to send
    array = np.random.rand(4, 4)
    socket.send(pickle.dumps(array))  # Serialize and send the array

    # Receive and deserialize the echoed array
    result = pickle.loads(socket.recv())
    print("📤 Sent:\n", array)
    print("📥 Received back:\n", result)


if __name__ == "__main__":
    main()
