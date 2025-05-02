# Joystick Control for Zima Robot

This document describes how to use a PS3 controller to drive the Zima robot.

## Overview

The joystick control system uses the ROS2 `joy` package to receive controller input and converts it to motor commands for the Zima robot. The controller can be used to drive the robot in differential drive mode, where the left joystick controls both forward/backward movement and turning.

## Hardware Requirements

- PS3 compatible controller (or knockoff)
- USB connection or Bluetooth capability
- Zima robot with ESP32 running zima_hardware_interface

## Default Mappings

### Button Mappings
- **X Button**: Button 0
- **Circle Button**: Button 1
- **Triangle Button**: Button 2
- **Square Button**: Button 3
- **L1 Button**: Button 4
- **R1 Button**: Button 5
- **L2 Button**: Button 6
- **R2 Button**: Button 7
- **Select Button**: Button 8
- **Start Button**: Button 9
- **PS Button**: Button 10
- **L3 Button** (Left Stick Press): Button 11
- **R3 Button** (Right Stick Press): Button 12
- **D-Pad Up**: Button 13
- **D-Pad Down**: Button 14
- **D-Pad Left**: Button 15
- **D-Pad Right**: Button 16

### Axis Mappings
- **Left Stick X-Axis**: Axis 0 (left/right)
- **Left Stick Y-Axis**: Axis 1 (up/down)
- **L2 Trigger**: Axis 2 (-1 when pressed, 1 when released)
- **Right Stick X-Axis**: Axis 3 (left/right)
- **Right Stick Y-Axis**: Axis 4 (up/down)
- **R2 Trigger**: Axis 5 (-1 when pressed, 1 when released)

### Control Functions
- **Left Joystick**: Control movement (forward/backward/turning)
- **R1 Button**: Enable motor control (must be held down)
- **R2 Button**: Emergency stop (all motors)
- **Select Button**: Cycle through control modes (drive, arm, custom1, custom2)

## Installation

1. Make sure the ROS2 joy package is installed:
   ```bash
   sudo apt-get update
   sudo apt-get install ros-humble-joy
   ```

2. Build the package:
   ```bash
   cd ~/Desktop/zima/src/ros2_ws
   colcon build --packages-select hardware_integration
   source install/setup.bash
   ```

## Usage

1. Connect your PS3 controller to your computer via USB or Bluetooth
2. Verify the controller is detected:
   ```bash
   ls -l /dev/input/js*
   ```

3. Test your controller mapping (optional but recommended):
   ```bash
   ros2 launch hardware_integration joy_tester.py
   ```
   Press buttons and move joysticks to see their corresponding indices displayed in the terminal.

4. Launch the joystick control system:
   ```bash
   ros2 launch hardware_integration joy_control.py
   ```
   
5. Control your robot:
   - Hold down R1 (button 5) to enable movement control
   - Use the left joystick to drive the robot
   - Press R2 (button 7) for emergency stop
   - Press Select (button 8) to cycle through control modes

## Parameters

The following parameters can be configured when launching:

- `joy_dev`: Joystick device path (default: `/dev/input/js0`)
- `joy_deadzone`: Deadzone for joystick inputs (default: `0.05`)
- `max_speed`: Maximum motor speed (default: `255`)
- `serial_port`: Serial port for hardware communication (default: `/dev/ttyUSB0`)
- `baud_rate`: Baud rate for serial communication (default: `115200`)

Example with custom parameters:
```bash
ros2 launch hardware_integration joy_control.py joy_dev:=/dev/input/js1 max_speed:=200
```

## Node Details

### joy_control_node

This node subscribes to `/joy` topic and publishes to `/motor_command` topic. It performs the following:

1. Reads joystick inputs for linear and angular movement
2. Applies deadzone and scaling
3. Calculates differential drive values for left and right motors
4. Publishes appropriate MotorCommand messages

The node only sends commands when the enable button (R1) is pressed, providing a safety feature.

## Troubleshooting

1. **Controller not detected**: 
   - Check the device path with `ls -l /dev/input/js*`
   - Update the joy_dev parameter with the correct path

2. **Motors not responding**:
   - Verify that you're holding the R1 button while moving the joystick
   - Check that the serial connection is working
   - Ensure the ESP32 is powered and running the correct firmware

3. **Unexpected movement**:
   - Adjust the deadzone parameter if the robot moves when the stick is at rest
   - Reduce max_speed if the robot moves too quickly
   - Check the orientation of the controller (joystick axes may be different)

## Control Modes

The joystick controller supports multiple control modes that can be cycled through using the Select button:

### 1. Drive Mode (Default)
- **Left Joystick**: Controls differential drive movement
  - Y-axis: Forward/backward movement
  - X-axis: Left/right turning
- **R1 Button**: Must be held to enable movement (safety feature)
- **R2 Button**: Emergency stop

### 2. Arm Mode
Reserved for future arm control implementation. When implemented, this mode will allow:
- Controlling servo positions for robotic arm joints
- Precise positional control
- Grip control

### 3. Custom Mode 1
Reserved for future custom control implementation.

### 4. Custom Mode 2
Reserved for future custom control implementation.

## Extensibility

The control system is designed to be easily extended. The codebase includes empty function placeholders for:

- `process_arm_commands()`: For controlling robot arm servos
- `process_custom1_commands()` & `process_custom2_commands()`: For custom control schemes
- `handle_dpad_inputs()`: For D-pad control functions
- `handle_shape_buttons()`: For X, Circle, Triangle, Square button actions
- `handle_trigger_buttons()`: For secondary L1/L2/R1/R2 functions
- `handle_thumbstick_buttons()`: For L3/R3 button press actions

To extend functionality, edit the joy_control.py file and implement these functions according to your needs.

These can be added to the joy_control.py file by adding new parameters and updating the joy_callback method.