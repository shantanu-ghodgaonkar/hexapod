from zeroconf import Zeroconf, ServiceBrowser
import socket
import zmq
import time
import msgpack
import numpy as np
from hexapod_v2_5_3 import hexapod

SERVICE_TYPE = "_mpcStep._tcp.local."


class ZeroconfListener:
    """
    Listener class for Zeroconf service discovery.
    Captures server IP and port when advertised.
    """

    def __init__(self):
        self.server_ip = None
        self.port = None

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            self.server_ip = socket.inet_ntoa(info.addresses[0])
            self.port = info.port
            print(f"Zeroconf discovered: {self.server_ip}:{self.port}")

    def remove_service(self, zeroconf, type, name):
        pass

    def update_service(self, zeroconf, type, name):
        pass


def discover_once(timeout=3):
    """
    Perform Zeroconf discovery once and return IP and port.
    """
    zeroconf = Zeroconf()
    listener = ZeroconfListener()
    browser = ServiceBrowser(zeroconf, SERVICE_TYPE, listener)
    start = time.time()
    while not listener.server_ip and (time.time() - start) < timeout:
        time.sleep(0.1)
    zeroconf.close()
    return listener.server_ip, listener.port


def create_socket(ip, port):
    """
    Create and connect a persistent ZMQ REQ socket.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{ip}:{port}")
    return socket


def send_array(socket, array):
    """
    Serialize and send NumPy array over ZMQ socket. Receive echo.
    """
    array = array.astype(np.float32)
    meta = {"shape": array.shape, "dtype": str(array.dtype)}
    packed = msgpack.packb({"meta": meta, "data": array.tolist()})
    socket.send(packed)
    response = msgpack.unpackb(socket.recv(), raw=False)
    result = np.array(response["data"], dtype=response["meta"]["dtype"])
    return result


def main():
    ip, port = discover_once()
    if not ip:
        print("Could not discover server")
        return

    socket = create_socket(ip, port)
    print(f"Connected to server at {ip}:{port}")

    # # Send multiple arrays
    # for i in range(10):
    #     array = np.random.rand(4, 4)
    #     t0 = time.perf_counter()
    #     result = send_array(socket, array)
    #     t1 = time.perf_counter()
    #     print(f"Round-trip: {(t1 - t0)*1000:.2f} ms")

    hexy = hexapod()
    # hexy.update_current_pose() using latest visual and joint data
    WAYPOINTS = 20
    horizon = 3
    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=1, leg_set=0)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            window += [wp[-1]] * (horizon - len(window))
        qi = send_array(socket=socket, array=np.concatenate(
            (hexy.qc, np.concatenate([w.flatten() for w in window]))))
        hexy.update_current_pose(q=qi)
        print(f'Res = {qi}')


def connect_socket():
    ip, port = discover_once()
    if not ip:
        print("Could not discover server")
        return

    socket = create_socket(ip, port)
    print(f"Connected to server at {ip}:{port}")
    return socket


if __name__ == "__main__":
    main()
