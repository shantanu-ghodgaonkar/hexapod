import re
import numpy as np

# Paste your data as a multi-line string
data = '''
#Name	motor_20
Mass	18E-03 Kg
Center of Mass	-205.601E-03 m, -118.741E-03 m, 93.594E-03 m
Moment of Inertia at Center of Mass   (Kg.m^2)
		Ixx	2035.055E-09
		Ixy	-304.957E-09
		Ixz	-543.384E-09
		Iyx	-304.957E-09
		Iyy	2397.169E-09
		Iyz	-310.556E-09
		Izx	-543.384E-09
		Izy	-310.556E-09
		Izz	2450.501E-09
#Name	tibia_
Mass	16.569E-03 Kg
Center of Mass	-219.88E-03 m, -126.948E-03 m, 54.01E-03 m
Moment of Inertia at Center of Mass   (Kg.m^2)
		Ixx	16582.375E-09
		Ixy	1299.124E-09
		Ixz	959.024E-09
		Iyx	1299.124E-09
		Iyy	15082.276E-09
		Iyz	553.693E-09
		Izx	959.024E-09
		Izy	553.693E-09
		Izz	3705.524E-09
'''

import re
import numpy as np

# Split the data into sections based on '#Name'
sections = data.strip().split('#Name')

# Initialize a list to store the results
objects = []

# Iterate over each section
for section in sections:
    section = section.strip()
    if not section:
        continue  # Skip empty sections
    lines = section.split('\n')
    # Initialize variables for this object
    obj = {}
    inertia_components = {}
    # Initialize units for this object
    inertia_units = 'Kg.m^2'  # Default units
    com_units = 'm'  # Default units
    # The first line is the object name
    obj['Name'] = lines[0].strip()
    # Now process the rest of the lines
    for line in lines[1:]:
        # Remove leading/trailing whitespace
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Extract the mass
        if line.startswith('Mass'):
            # Extract the number before 'Kg'
            match = re.search(r'Mass\s+([-\d.Ee]+)\s*Kg', line)
            if match:
                mass_str = match.group(1)
                obj['Mass'] = float(mass_str)
        # Extract the Center of Mass
        elif line.startswith('Center of Mass'):
            # Extract units from the line
            com_units_matches = re.findall(r'\s*m', line)
            if len(com_units_matches) == 0:
                com_units_matches = re.findall(r'\s*mm', line)
                com_units = 'mm'
            else:
                com_units = 'm'
            # Extract the numbers and units
            matches = re.findall(r'([-\d.Ee]+)\s*(mm|m)', line)
            if len(matches) == 3:
                com = []
                for num_str, unit in matches:
                    num = float(num_str)
                    if unit == 'mm':
                        num /= 1000.0  # Convert mm to m
                    # If unit is 'm', no conversion needed
                    com.append(num)
                obj['COM'] = np.array(com)
        # Extract the units for the Moment of Inertia
        elif 'Moment of Inertia at Center of Mass' in line:
            # Extract units, which are in parentheses
            match = re.search(r'\(([^)]+)\)', line)
            if match:
                inertia_units = match.group(1)
        # Extract inertia components
        elif line.startswith('I'):
            # Split the line into key and value
            key_value = re.split(r'\s+', line)
            if len(key_value) == 2:
                key, value_str = key_value
                # Convert value to float
                value = float(value_str)
                # Convert value to kg·m² if necessary
                if inertia_units == 'g mm^2':
                    value *= 1e-9  # Convert from (g mm^2) to (kg m^2)
                elif inertia_units == 'Kg.m^2':
                    pass  # No conversion needed
                else:
                    print(f"Unknown inertia units: {inertia_units}")
                inertia_components[key] = value
        # Skip other lines
        else:
            continue
    # Map the inertia components to matrix indices
    index_map = {
        'Ixx': (0, 0),
        'Ixy': (0, 1),
        'Ixz': (0, 2),
        'Iyx': (1, 0),
        'Iyy': (1, 1),
        'Iyz': (1, 2),
        'Izx': (2, 0),
        'Izy': (2, 1),
        'Izz': (2, 2)
    }
    # Initialize the inertia matrix
    I = np.zeros((3, 3))
    # Fill the inertia matrix with the components
    for key, value in inertia_components.items():
        if key in index_map:
            i, j = index_map[key]
            I[i, j] = value
    # Since the inertia matrix is symmetric, mirror the off-diagonal elements
    for i in range(3):
        for j in range(3):
            if i != j:
                I[j, i] = I[i, j]
    obj['InertiaMatrix'] = I
    # Append the object data to the list
    objects.append(obj)

# Function to compute the total mass
def compute_total_mass():
    total_mass = 0
    for obj in objects:
        mass = obj.get('Mass', 0)
        total_mass += mass
    return total_mass

# Function to compute the combined center of mass
def compute_combined_com():
    total_mass = compute_total_mass()
    if total_mass == 0:
        return np.array([0, 0, 0])
    combined_com = np.zeros(3)
    for obj in objects:
        mass = obj.get('Mass', 0)
        com = obj.get('COM', np.zeros(3))
        combined_com += mass * com
    combined_com /= total_mass
    return combined_com

# Compute and print the combined mass
combined_mass = compute_total_mass()
print(f"Combined Mass: {combined_mass} Kg")

# Compute and print the combined center of mass
combined_com = compute_combined_com()
print(f"Combined Center of Mass: {combined_com}")

# Compute the combined inertia tensor
def compute_combined_inertia():
    combined_inertia = np.zeros((3, 3))
    for obj in objects:
        mass = obj.get('Mass', 0)
        com = obj.get('COM', np.zeros(3))
        inertia = obj.get('InertiaMatrix', np.zeros((3, 3)))
        # Compute displacement vector from object's COM to combined COM
        d = com - combined_com
        # Use the parallel axis theorem
        inertia_shifted = inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        # Add to the combined inertia
        combined_inertia += inertia_shifted
    return combined_inertia

# Compute and print the combined inertia tensor
combined_inertia = compute_combined_inertia()
print("Combined Inertia Matrix with respect to the combined Center of Mass:")
print(combined_inertia)
