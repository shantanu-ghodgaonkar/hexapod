from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

# Step 1: Connect to the CoppeliaSim Remote API server
client = RemoteAPIClient()
# sim = client.get_object('sim')
sim = client.getObject('sim')

# Step 2: Start the simulation
print("Starting the simulation...")
sim.startSimulation()

# Step 3: Pause for 5 seconds to let the simulation run
time.sleep(5)

# Step 4: Stop the simulation
print("Stopping the simulation...")
sim.stopSimulation()
