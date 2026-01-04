# Zima PS3 Teleoperator

PS3 controller teleoperation interface for the Zima robot, compatible with the LeRobot framework.

## Installation

```bash
cd lerobot_teleoperator_zima_ps3_teleop
pip install -e .
```

## Usage

### With LeRobot CLI

```bash
lerobot-teleoperate \
    --robot.type=zima_robot \
    --robot.address=10.0.0.100 \
    --robot.port=5000 \
    --teleop.type=zima_ps3_teleop
```

### Control Scheme

The control scheme matches the ROS2 joy_control node exactly:

#### Track Control (D-Pad)
- **D-Pad Up**: Move forward
- **D-Pad Down**: Move backward
- **D-Pad Left**: Turn left
- **D-Pad Right**: Turn right

#### Arm Control (Analog Sticks)
- **Left Stick X-axis**: Arm position Y (left/right)
- **Left Stick Y-axis**: Arm position X (forward/back)
- **Right Stick X-axis**: Arm rotation X (roll, inverted)
- **Right Stick Y-axis**: Arm position Z (up/down)

#### Gripper Control
- **L2 Trigger**: Close gripper
- **R2 Trigger**: Open gripper

## Configuration Parameters

- `deadzone`: Analog stick deadzone (default: 0.15)
- `max_speed`: Maximum track speed multiplier (default: 1.0)
- `max_arm_delta`: Maximum arm position delta (default: 1.0)
- `max_rotation_delta`: Maximum arm rotation delta (default: 1.0)

## Requirements

- Python 3.10+
- LeRobot framework
- pygame (for gamepad support)
- PS3 controller (or compatible gamepad)
