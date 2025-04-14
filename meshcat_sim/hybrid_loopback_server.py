from zeroconf import ServiceInfo, Zeroconf
import socket
import zmq
import pickle
import threading

# Configuration for ZeroMQ and Zeroconf
ZMQ_PORT = 5555
SERVICE_TYPE = "_loopback._tcp.local."
SERVICE_NAME = "LoopbackService"


def get_real_ip():
    """
    Gets the actual IP address of the host on the local network,
    avoiding loopback addresses like 127.0.1.1.

    Returns:
        str: The local network IP address.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def advertise_service():
    """
    Advertises the loopback echo service on the local network using Zeroconf.

    Returns:
        Zeroconf: The active Zeroconf instance (keep it alive to maintain registration).
    """
    local_ip = get_real_ip()
    hostname = socket.gethostname()
    info = ServiceInfo(
        SERVICE_TYPE,
        f"{SERVICE_NAME}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(local_ip)],
        port=ZMQ_PORT,
        properties={},
        server=f"{hostname}.local.",
    )
    zeroconf = Zeroconf()
    zeroconf.register_service(info)
    print(f"📱 Service advertised as {SERVICE_NAME} on {local_ip}:{ZMQ_PORT}")
    return zeroconf


def start_udp_discovery_listener():
    """
    Starts a lightweight UDP responder that listens for broadcast messages
    from clients seeking the loopback service. Replies with a confirmation.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 50000))
    print("🔍 UDP discovery responder listening...")
    while True:
        data, addr = sock.recvfrom(1024)
        if data == b"where_is_loopback":
            print(f"📨 Discovery ping from {addr}")
            sock.sendto(b"loopback_here", addr)


def start_zmq_loopback():
    """
    Starts the ZeroMQ REP (reply) server to handle echo requests.
    Waits for a NumPy array from the client, prints it, then echoes it back.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"⚙️ ZMQ loopback server listening on port {ZMQ_PORT}...")

    while True:
        data = socket.recv()
        array = pickle.loads(data)
        print("📥 Received array:\n", array)
        socket.send(pickle.dumps(array))  # Echo the array back


if __name__ == "__main__":
    """
    Entry point for the loopback server.
    - Advertises itself with Zeroconf
    - Listens for UDP broadcast-based discovery
    - Handles ZMQ echo requests
    """
    zeroconf = advertise_service()
    threading.Thread(target=start_udp_discovery_listener, daemon=True).start()
    try:
        start_zmq_loopback()
    finally:
        zeroconf.unregister_all_services()
        zeroconf.close()
