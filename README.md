# Zima: End-to-End Control Platform for Robotic Manipulation

Zima is a low-cost rover with a 5DOF arm. Designed as a personal research platform for developing and testing end-to-end neural networks in robotic manipulation and navigation tasks.

<img src="https://github.com/user-attachments/assets/a8970ae2-b656-4fd7-85b0-6ae1a3be184e" width="400">

## Current Status
- ✅ Hardware prototype operational (5-DOF arm + differential drive base)
- ✅ ROS2 control stack deployed on Raspberry Pi 5
- ✅ MuJoCo simulation with teleoperation and data collection
- 🔄 Training vision-based navigation policies (ResNet→action network)

## Repository Structure
- `src/imitation_learning` - Model definitions and training scripts, Mujoco simulation
- `src/zima_ros2/` - ROS2 hardware interface
- `src/firmware/` - ESP32 motor control firmware

## Hardware
- 5-DOF custom robotic arm (4x MG996R servo joints, 1x gripper)
- Differential drive base (modified Recon 6.0 Rover chassis)
- Dual camera setup: wrist-mounted + front-facing RGB
- ESP32 for low-level motor control
- Raspberry Pi 5 for vision processing and policy execution

## Tech Stack
ROS2 Jazzy | PyTorch | MuJoCo | ESP-IDF

