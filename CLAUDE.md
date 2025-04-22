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
- Simulation testing (MuJoCo, Isaac Sim)

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
- Utilizing PI_0 model from Physical Intelligence

### Current Development Focus
- Hardware Integration
  * Raspberry Pi 5 as main computer
  * ESP32 for hardware interface
  * Rotary encoder testing
  * Motor control validation
  * Comprehensive sensor integration (camera, microphone)
  * Startup behavior optimization
  * Track mount design and fabrication

- Software Research
  * PI_0 model adaptation
  * Vision Language Action (VLA) implementation
  * Precise 3D positioning system
  * Data collection methodology using PS3 controller
  * Innovative memory representation (directed graph model)
  * Student-teacher machine learning approach
  * Local memory search algorithms

### Computational Infrastructure
- Local Compute: Raspberry Pi 5
- Cloud Training: Vast.ai (H100 GPU)
- Model Deployment: AWS Instances
  * G6 Instance Types (Verified):
    - g6.xlarge: 16GB RAM, 1 GPU (24GB)
    - g6.2xlarge: 32GB RAM, 1 GPU (24GB)
    - g6.4xlarge: 64GB RAM, 1 GPU (24GB)
    - g6.8xlarge: 128GB RAM, 1 GPU (24GB)
    - g6.16xlarge: 256GB RAM, 1 GPU (24GB)
    - g6.12xlarge: 192GB RAM, 4 GPUs (96GB)
    - g6.24xlarge: 384GB RAM, 4 GPUs (96GB)
    - g6.48xlarge: 768GB RAM, 8 GPUs (192GB)
  * G6e Instance Types (Tensor Core GPUs):
    - g6e.xlarge: 32GB RAM, 1 GPU (48GB)
    - g6e.2xlarge: 64GB RAM, 1 GPU (48GB)
    - g6e.4xlarge: 128GB RAM, 1 GPU (48GB)
    - g6e.8xlarge: 256GB RAM, 1 GPU (48GB)
    - g6e.16xlarge: 512GB RAM, 1 GPU (48GB)
    - g6e.12xlarge: 384GB RAM, 4 GPUs (192GB)
    - g6e.24xlarge: 768GB RAM, 4 GPUs (192GB)
    - g6e.48xlarge: 1536GB RAM, 8 GPUs (384GB)

### Project Documentation and Synchronization
⚠️ CRITICAL WORKFLOW REQUIREMENT ⚠️
- ALWAYS run `keep_sync.py` before starting work or making significant changes
- Purpose: Synchronize latest project details, notes, and research annotations
- Synchronization Checklist:
  * Verify Google Keep notes are up-to-date
  * Check for any new research insights
  * Update local documentation with latest findings
  * Confirm no conflicting information exists
- Sync Script Location: `./keep_sync.py`
- Frequency: 
  * Before each work session
  * After major project milestones
  * When switching between different project components

🚨 FAILURE TO SYNC MAY RESULT IN:
- Outdated project information
- Missed critical research updates
- Potential project direction misalignment

### Development Roadmap
1. Precise 3D Positioning
2. Comprehensive Data Collection
3. VLA Model Fine-Tuning
4. Hierarchical Decision-Making
5. Adaptive Memory System

### Technical Stack
- ROS2 Framework
- Arduino/ESP32 Microcontroller
- URDF Robotics Modeling
- Python and C++ Development
- PI_0 Vision Language Action Model

### Research Objectives
- Precise robotic manipulation
- Edge computing for robotics
- Adaptive learning techniques
- Intelligent sensor data interpretation

## Key Constraints
- Arm Length: 30cm (Verified)
- Optimize for Raspberry Pi 5 computational limits
- Emphasize precise grip capabilities
- Meet complex object manipulation requirements