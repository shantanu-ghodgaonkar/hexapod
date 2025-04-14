# from zeroconf import ServiceInfo, Zeroconf
# import socket
# import zmq
# import json
# import numpy as np
# from scipy.optimize import minimize


# def cost_function(x, *args):
#     # Replace with your actual cost function.
#     return np.sum((x - np.array(args))**2)


# def advertise_service(port=5555, service_name="MinimizationService"):
#     # Get local IP address
#     hostname = socket.gethostname()
#     local_ip = socket.gethostbyname(hostname)

#     # Define service info
#     service_type = "_minimization._tcp.local."
#     service_info = ServiceInfo(
#         service_type,
#         f"{service_name}.{service_type}",
#         addresses=[socket.inet_aton(local_ip)],
#         port=port,
#         properties={},
#         server=f"{hostname}.local.",
#     )

#     zeroconf = Zeroconf()
#     zeroconf.register_service(service_info)
#     print(f"Service '{service_name}' advertised on {local_ip}:{port}")
#     return zeroconf  # Keep reference to avoid premature closing


# def main():
#     # Advertise the service
#     zeroconf = advertise_service(port=5555, service_name="MinimizationService")

#     # Create a ZeroMQ REP socket.
#     context = zmq.Context()
#     socket = context.socket(zmq.REP)
#     socket.bind("tcp://*:5555")
#     print("Minimization server is listening on port 5555...")

#     try:
#         while True:
#             try:
#                 message = socket.recv()
#                 data = json.loads(message.decode('utf-8'))

#                 # Expecting 'initial_guess' and 'params' in the request
#                 initial_guess = np.array(data.get('initial_guess'))
#                 params = data.get('params', [])
#                 print(
#                     f"Received request with initial guess: {initial_guess} and params: {params}")

#                 # Run the minimization
#                 result = minimize(
#                     cost_function, initial_guess, args=tuple(params))

#                 response = {
#                     'x': result.x.tolist(),
#                     'fun': result.fun,
#                     'success': result.success,
#                     'message': result.message
#                 }
#                 socket.send_string(json.dumps(response))
#                 print("Result sent back to client.")
#             except Exception as e:
#                 error_response = json.dumps({'error': str(e)})
#                 socket.send_string(error_response)
#                 print("Error processing request:", e)
#     finally:
#         zeroconf.unregister_all_services()
#         zeroconf.close()


# if __name__ == '__main__':
#     main()


#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print(f"Received request: {message}")

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")
