# Zima: Low-Cost Imitation Learning Robot

Zima is a custom-built rover with a 5-DOF arm, designed as a personal research platform for imitation learning in robotic manipulation. It uses teleoperated demonstrations to train vision-based policies that map camera observations directly to motor actions.

[Devlog](https://eryk.ca/zima_devlog.html) | [Dataset](https://huggingface.co/datasets/ehalicki/zima-rubiks-cube)

<img src="https://github.com/user-attachments/assets/a8970ae2-b656-4fd7-85b0-6ae1a3be184e" width="400">

## Current Status
- Hardware prototype operational (5-DOF arm + tracked differential drive base)
- ROS2 control stack deployed on Raspberry Pi 5
- LeRobot integration for teleoperation, dataset recording, and training
- Training and evaluating manipulation policies (ACT, VLAs, world action models, etc.)
- Datasets hosted on HuggingFace with online visualization

## Hardware
- **Arm**: 4x MG996R servo joints + 1x gripper servo, custom forward/inverse kinematics solver
- **Base**: Modified Recon 6.0 Rover chassis, 2x DC motors with rotary encoders
- **Cameras**: Wrist-mounted + front-facing USB RGB (480x640)
- **Compute**: Raspberry Pi 5 (vision + policy execution), ESP32 (hardware interfacing)
- **CAD**: Custom 3D-printed parts designed in FreeCAD (`cad_models/`)

## Repository Structure
```
src/
  firmware/                              # ESP32 motor/servo control (Arduino)
  zima_ros2/
    zima_controls/                       # Arm kinematics solver & controller
    zima_hardware_integration/           # Serial, TCP, camera, and joystick interfaces
    zima_launch/                         # ROS2 launch configurations
    zima_msgs/                           # Custom message types (ArmStateDelta, MotorCommand, etc.)
  lerobot_robot_zima_robot/              # LeRobot Robot interface (TCP to hardware)
  lerobot_teleoperator_zima_ps3_teleop/  # PS3 gamepad teleoperator plugin
  scripts/                              # Bash scripts for recording, training, teleoperation
  legacy/                               # Deprecated ML/simulation code
cad_models/                             # FreeCAD design files
docs/                                   # Circuit diagrams
```

## Architecture

```
PS3 Controller ──► LeRobot Teleoperator ──► TCP ──► ROS2 (Pi 5) ──► Serial ──► ESP32 ──► Motors/Servos
                                                        │
                                                   Cameras ──► Observations
                                                        │
                                              LeRobot Dataset ──► HuggingFace Hub
                                                        │
                                                 Policy Training ──► Deployment
```

**Control flow**: The ESP32 handles low-level servo/motor control over serial. ROS2 nodes on the Pi manage kinematics, camera streaming, and a TCP server (port 5000) that bridges LeRobot to the hardware. The LeRobot framework handles teleoperation input, dataset recording, policy training, and inference.

**Kinematics**: Custom solver using YAML-defined link parameters, numerical IK with KD-tree initialization, and safety collision checks against predefined bounding boxes. Configuration in `src/zima_ros2/src/zima_controls/config/`.

## Usage

Teleoperate the robot:
```bash
lerobot-teleoperate --robot.type=zima_robot --robot.address=192.168.2.5
```

Record a dataset:
```bash
lerobot-record --robot.type=zima_robot --teleop.type=zima_ps3_teleop \
  --dataset.repo_id=ehalicki/zima-rubiks-cube --dataset.num_episodes=10
```

Train a policy:
```bash
lerobot-train --dataset.repo_id=ehalicki/zima-rubiks-cube \
  --policy.type=act --policy.device=mps
```
