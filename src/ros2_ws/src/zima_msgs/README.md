# Zima Messages Package

This package defines custom ROS2 message types for the Zima robot hardware interface.

## Message Types

### ServoCommand

Command for a single servo motor:
- `std_msgs/Header header`: Standard ROS header
- `uint8 servo_id`: ID of the servo (0-7)
- `float32 position`: Position in degrees (0-180)

### MotorCommand

Command for DC motors:
- `std_msgs/Header header`: Standard ROS header
- `uint8 motor_id`: Which motor (MOTOR_LEFT=0, MOTOR_RIGHT=1)
- `uint8 action`: Action (ACTION_FORWARD=0, ACTION_REVERSE=1, ACTION_STOP=2)
- `int32 speed`: Speed value (0-255 for forward/reverse, 0 or 1 for stop)

### EncoderData

Data from wheel encoders:
- `std_msgs/Header header`: Standard ROS header
- `int32 left_clicks`: Encoder clicks for left wheel
- `int32 right_clicks`: Encoder clicks for right wheel

### ServoPositions

Current positions of all servo motors:
- `std_msgs/Header header`: Standard ROS header
- `float32[] positions`: Current positions of all servos (array of 8 values)

### HardwareCommand

Low-level command for any hardware subsystem:
- `std_msgs/Header header`: Standard ROS header
- `uint8 subsystem`: The subsystem ID (0-13)
- `float32 value`: The command value

Subsystem mapping:
- 0-7: Servo motors (position in degrees)
- 8: Left DC Motor Forward (speed 0-255)
- 9: Left DC Motor Reverse (speed 0-255)
- 10: Left DC Motor Stop (0 or 1)
- 11: Right DC Motor Forward (speed 0-255)
- 12: Right DC Motor Reverse (speed 0-255)
- 13: Right DC Motor Stop (0 or 1)

### HardwareStatus

Status data from all hardware subsystems:
- `std_msgs/Header header`: Standard ROS header
- `int32 clicks_left`: Left encoder clicks
- `int32 clicks_right`: Right encoder clicks
- `float32[] servo_positions`: Array of 8 servo positions

## Usage

To use these messages in your ROS2 package, add a dependency in your `package.xml`:

```xml
<depend>zima_msgs</depend>
```

And include the message headers in your C++ code:

```cpp
#include "zima_msgs/msg/servo_command.hpp"
#include "zima_msgs/msg/motor_command.hpp"
// etc.
```

Or import them in your Python code:

```python
from zima_msgs.msg import ServoCommand, MotorCommand
# etc.
```