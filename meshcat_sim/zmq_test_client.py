
# from zeroconf import ServiceBrowser, Zeroconf
# import socket
# import time
# import zmq
# import json
# import numpy as np


# class MyListener:
#     def __init__(self):
#         self.server_ip = None
#         self.port = None

#     def remove_service(self, zeroconf, type, name):
#         pass

#     def add_service(self, zeroconf, type, name):
#         info = zeroconf.get_service_info(type, name)
#         if info:
#             self.server_ip = socket.inet_ntoa(info.addresses[0])
#             self.port = info.port
#             print(f"Discovered service '{name}': {self.server_ip}:{self.port}")


# def discover_service(timeout=5, service_type="_minimization._tcp.local."):
#     zeroconf = Zeroconf()
#     listener = MyListener()
#     browser = ServiceBrowser(zeroconf, service_type, listener)
#     # Wait for service to be discovered.
#     start = time.time()
#     while listener.server_ip is None and (time.time() - start) < timeout:
#         time.sleep(0.1)
#     zeroconf.close()
#     if listener.server_ip:
#         return listener.server_ip, listener.port
#     else:
#         return None, None


# def offload_minimization(initial_guess, params):
#     # Discover the server
#     server_ip, port = discover_service()
#     if not server_ip:
#         raise Exception("Service not found on the network.")
#     print(f"Connecting to server at {server_ip}:{port}")

#     # Create ZeroMQ REQ socket.
#     context = zmq.Context()
#     socket = context.socket(zmq.REQ)
#     socket.connect(f"tcp://{server_ip}:{port}")

#     # Prepare the request data.
#     data = {
#         "initial_guess": initial_guess.tolist() if isinstance(initial_guess, np.ndarray) else initial_guess,
#         "params": params
#     }
#     message = json.dumps(data)
#     socket.send_string(message)

#     # Wait for response.
#     response_message = socket.recv()
#     response = json.loads(response_message.decode('utf-8'))
#     return response


# # Example usage:
# if __name__ == "__main__":
#     initial_guess = np.array([0.0, 0.0, 0.0])
#     params = [1.0, 2.0, 3.0]
#     try:
#         result = offload_minimization(initial_guess, params)
#         print("Optimization result:", result)
#     except Exception as e:
#         print("Failed to offload minimization:", e)

#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print(f"Sending request {request} …")
    socket.send(b"Hello")

    #  Get the reply.
    message = socket.recv()
    print(f"Received reply {request} [ {message} ]")
