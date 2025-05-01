# Zima Hardware Interface

## Overview

The Zima Hardware Interface is a modular Arduino sketch designed to control multiple hardware components for the Zima robotic platform, including:
- Servo Motors
- Rotary Encoders
- DC Motors

## Hardware Components

### Servo Motors
- Total servos: 8
- Configurable pin assignments
- Smooth position control
- Special group handling for synchronized movements

### Rotary Encoders
- Two independent encoders
- Interrupt-based click tracking
- Debounce protection

### DC Motors
- Single DC motor control
- Forward/Reverse/Stop functionality
- Configurable speed and stop mode

## Serial Communication Protocol

### Initialization
- Baud Rate: 115200
- Initialization Message: `ZIMA:INIT:HARDWARE_INTERFACE`

### Command Format
Commands follow the pattern: `<subsystem><value>`

#### Servo Motor Control (0-7)
- Subsystem: 0-7 (corresponding to servo index)
- Value: Angle in degrees (-1 to disable, 0-180 for position)
- Examples:
  - `2<45` - Move servo 2 to 45 degrees
  - `1<-1` - Disable servo 1

#### DC Motor Control
- Forward (8): `8<speed>` (0-255)
  - Example: `8<200` - Move DC motor forward at 200 speed
- Reverse (9): `9<speed>` (0-255)
  - Example: `9<150` - Move DC motor in reverse at 150 speed
- Stop (10): `10<mode>`
  - `10<1>` - Soft stop
  - `10<0>` - Hard brake

### Status Reporting
Status messages sent in format: `ZIMA:DATA:CLICKSLEFT=X,CLICKSRIGHT=Y,SERVOS=a|b|c|d|e|f|g|h`

- `CLICKSLEFT/RIGHT`: Encoder click counts
- `SERVOS`: Current servo positions (in order)

## Configuration

Edit `config.h` to modify:
- Pin assignments
- Encoder settings
- Serial communication parameters
- Motor control configurations

## Code Structure

- `config.h`: Centralized configuration
- `servo_control.h`: Servo motor management
- `encoder_control.h`: Encoder tracking
- `dcmotor_control.h`: DC motor control
- `zima_hardware_interface.ino`: Main interface logic

## Usage Notes

1. Initialize serial communication at 115200 baud
2. Send commands as described in the protocol
3. Parse incoming status messages
4. Maintain a short delay between commands

## Error Handling

- Invalid subsystem/command: Silently ignored
- Out-of-range values: Clamped or ignored
- Serial communication: Robust parsing with end marker

## Future Improvements

- Add error reporting
- Implement more complex motion profiles
- Support for additional motor types
- Enhanced synchronization between components

## Dependencies

- ESP32Servo library
- ESP32MX1508 library

## Troubleshooting

- Verify serial connection
- Check pin configurations
- Ensure proper power supply
- Monitor serial output for initialization messages

---

*Generated for Zima Robotics Platform*