import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the parametric functions


def x(t):
    return ((1 - t) * 0) + (t * (-0.0125))


def y(t):
    return ((1 - t) * (-0.25)) + (t * (- 0.2283))


def z(t):
    # return (-5 * ((t - 0.5) ** 2)) + 0.025
    ss = 0.025
    b = ss / 0.25
    a = -b
    return (a * (t**2)) + (b * t)


# Generate t values
t_values = np.linspace(0, 1, 400)

# Calculate x, y, and z values
x_values = x(t_values)
y_values = y(t_values)
z_values = z(t_values)

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the parabola in 3D space
ax.plot(x_values, y_values, z_values, label='3D Parabola', color='teal')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Plot of the Parabola')
ax.legend()

# Show the plot
plt.show()
