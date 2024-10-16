# Hexapod Robot

## Project Overview

This project involves the manufacturing and development of an all-terrain hexapod robot, primarily designed for **search and rescue operations**. The robot will be capable of traversing rough and uneven terrain, providing versatility and stability. 

### Key Features:
- **Transformable Leg Configuration**: Two of the robot’s six legs will be multi-functional. In addition to functioning as legs, they will be equipped with additional tools (such as grippers, TBD) that enable the robot to switch between a 6-leg and 4-leg configuration. This allows the robot to perform tasks beyond simple locomotion, making it a highly adaptable tool in rescue scenarios.
  
- **Simulation**: The kinematics of the robot are being simulated using **Pinocchio Rigid Body Dynamics**, with **SciPy** used for optimization. **Meshcat** is used for visualizing the robot's movement.

- **Hardware Development**: 
  - The 6 legs of the robot have been 3D printed.
  - Future development includes the acquisition of 18 motors (3 per leg) to complete the hardware assembly.

- **Planned Features**:
  - **Autonomy**: SLAM (Simultaneous Localization and Mapping) will be implemented for autonomous navigation in unknown environments.
  - **Optimal Control**: MPC (Model Predictive Control) will be integrated for controlling the robot's movement in an optimal way.

## Repository Structure

This repository currently contains the following:
- URDF file for the hexapod robot’s structure.
- CAD files in **.itp format** for the robot's physical design.
- Initial setup for simulations using Pinocchio and SciPy.
- **Books and resources** providing background knowledge for the development of control algorithms.

Further development is ongoing.

## Future Development

- **Integration of SLAM** for autonomous navigation.
- **MPC** for optimal control.
- Further development of the robot’s hardware and software, including the addition of multi-functional tools on two of the legs.

## License

This project does not currently have an associated license.
