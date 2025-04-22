# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

### ROS2 Workspace
- Build: `colcon build`
- Test: `colcon test`
- Single Package Test: `colcon test --packages-select <package_name>`
- Lint: `ament_flake8`, `ament_cpplint`, `ament_copyright`

### Arduino/ESP32
- Compile: Arduino IDE or PlatformIO
- Upload: Arduino IDE 'Upload' button
- Serial Monitor: 115200 baud rate

## Code Style Guidelines

### Python (ROS2)
- Follow PEP 8 style guide
- Use type hints
- Max line length: 120 characters
- Imports: Grouped (standard, third-party, local)
- Error Handling: Use try-except, log errors
- Naming: 
  * snake_case for functions, variables
  * UPPER_CASE for constants
  * CamelCase for classes

### C++ (ROS2)
- C++17 standard
- Use modern C++ features
- Compile with `-Wall -Wextra -Wpedantic`
- Namespace usage
- RAII principle
- Smart pointer usage

### Arduino/ESP32
- Camel case for functions
- ALL_CAPS for constants
- Minimize global variables
- Use `#define` for pin definitions
- Handle serial communication robustly
- Implement error checking

## Project-Specific Guidelines
- Modular design
- Separate hardware abstraction
- Consistent error logging
- Document complex logic
- Memory efficiency (esp. for microcontrollers)

## Testing Strategy
- Unit tests for each component
- Integration tests for ROS2 nodes
- Hardware-in-the-loop testing
- Continuous integration recommended

## Performance Considerations
- Optimize for low-latency systems
- Minimize dynamic memory allocation
- Use interrupts where possible
- Profile code regularly

## Project Zima Specific Details
### System Overview
- Articulated robotic arm with multi-joint design
- Tracked mobile base
- Research platform for Vision Language Action (VLA) models

### Current Development Focus
- Hardware Integration
  * Rotary encoder testing
  * Motor control validation
  * Startup behavior optimization
  * Track mount design and fabrication

- Software Research
  * Innovative memory representation (directed graph model)
  * Student-teacher machine learning approach
  * Local memory search algorithms

### Technical Stack
- ROS2 Framework
- Arduino/ESP32 Microcontroller
- URDF Robotics Modeling
- Python and C++ Development