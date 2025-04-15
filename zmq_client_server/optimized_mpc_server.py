from typing import List
from zeroconf import ServiceInfo, Zeroconf
import socket
import zmq
import msgpack
import numpy as np
import threading
from hexapod_v2_5_3 import hexapod

ZMQ_PORT = 5555
SERVICE_TYPE = "_mpcStep._tcp.local."
SERVICE_NAME = "MPCService"


def step(window: List[np.ndarray], hexy: hexapod, q: np.ndarray):
    hexy.update_current_pose(q=q)
    q_i = hexy.mpc_step(current_q=hexy.qc,
                        desired_seq=window, horizon=len(window))
    return q_i[7:]


def get_real_ip():
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
    print(f"Service advertised as {SERVICE_NAME} on {local_ip}:{ZMQ_PORT}")
    return zeroconf


def start_udp_discovery_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 50000))
    print("UDP discovery responder listening...")
    while True:
        data, addr = sock.recvfrom(1024)
        if data == b"where_is_loopback":
            print(f"Discovery ping from {addr}")
            sock.sendto(b"loopback_here", addr)


def start_zmq(hexy: hexapod):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # socket.setsockopt(zmq.TCP_NODELAY, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"ZMQ MPC server listening on port {ZMQ_PORT}...")

    while True:
        data = socket.recv()
        message = msgpack.unpackb(data, raw=False)
        meta = message["meta"]
        array = np.array(message["data"], dtype=meta["dtype"])
        print("Received array:\n", array)
        if (array.shape[0] - 25) % 21 != 0:
            raise ValueError(
                f'Received array of shape {array.shape}, but expected to be (25 + 21*n, )')
        q = array[:25]
        window = [array[i:i+21].reshape(21, 1)
                  for i in range(25, array.shape[0], 21)]
        res = step(window=window, hexy=hexy, q=q)

        reply = msgpack.packb({"meta": meta, "data": res})
        socket.send(reply)


if __name__ == "__main__":
    hexy = hexapod()
    zeroconf = advertise_service()
    threading.Thread(target=start_udp_discovery_listener, daemon=True).start()
    try:
        start_zmq(hexy=hexy)
    finally:
        zeroconf.unregister_all_services()
        zeroconf.close()
