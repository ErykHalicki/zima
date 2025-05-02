# Hardware Integration Package

This package provides a ROS2 interface for the Zima robot hardware. It handles serial communication with the ESP32 microcontroller, converting between ROS2 messages and the serial command format used by the hardware interface.

## Architecture

The hardware integration is split into three nodes:

1. **Serial Receiver Node** (`serial_receiver`): Reads data from the serial port and publishes it as ROS2 messages
2. **Serial Sender Node** (`serial_sender`): Receives hardware commands and sends them to the hardware via serial port
3. **Command Generator Node** (`command_generator`): Converts high-level ROS2 messages (e.g., JointState) to hardware commands

## Message Types

The hardware integration uses custom message types defined in the `zima_msgs` package:

- `ServoCommand`: Command for a single servo motor
- `MotorCommand`: Command for DC motors (forward, reverse, stop)
- `EncoderData`: Data from wheel encoders
- `ServoPositions`: Current positions of all servo motors
- `HardwareCommand`: Low-level command for any hardware subsystem
- `HardwareStatus`: Status data from all hardware subsystems

## Usage

### Launch the Complete System

```bash
ros2 launch hardware_integration hardware_interface.py
```

### Launch with Custom Parameters

```bash
ros2 launch hardware_integration hardware_interface.py serial_port:=/dev/ttyACM0 baud_rate:=115200 update_rate:=100.0
```

### Running Individual Nodes

```bash
# Run the serial receiver node
ros2 run hardware_integration serial_receiver

# Run the serial sender node
ros2 run hardware_integration serial_sender

# Run the command generator node
ros2 run hardware_integration command_generator
```

## Topics

### Published Topics

- `/hardware_status` (zima_msgs/HardwareStatus): Complete hardware status
- `/encoder_data` (zima_msgs/EncoderData): Encoder data from wheels
- `/servo_positions` (zima_msgs/ServoPositions): Current servo positions

### Subscribed Topics

- `/joint_states` (sensor_msgs/JointState): Joint state commands for servos
- `/servo_command` (zima_msgs/ServoCommand): Direct servo commands
- `/motor_command` (zima_msgs/MotorCommand): DC motor commands
- `/hardware_command` (zima_msgs/HardwareCommand): Low-level hardware commands

## Parameters

- `serial_port` (string, default: '/dev/ttyUSB0'): Serial port for hardware communication
- `baud_rate` (int, default: 115200): Baud rate for serial communication
- `timeout` (double, default: 0.1): Serial read/write timeout in seconds
- `update_rate` (double, default: 50.0): Rate in Hz for reading from the serial port