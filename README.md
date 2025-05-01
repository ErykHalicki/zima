# Zima: End to End Robotic Manipulation Platform

## Project Overview

Zima is a low cost autonomous robotic rover designed as a research platform for developing and testing end-to-end neural networks in robotic manipulation tasks. 

## Current Development Focus
- PI_0 VLA model adaptation
- Nueral Network based Inverse Kinematics
- Long term memory representation using directed graph model

## Current System Architecture

### Hardware Components
- Custom Articulated robotic arm
- Tracked mobile base (hacked from Recon 6.0 Rover Toy)
- ESP32 microcontroller
- Raspberry Pi 5 as primary computer

### Software Stack
- ROS2 framework
- PyTorch
- Custom hardware interface

## Computational Infrastructure
- Local Compute: Raspberry Pi 5 + ESP32
- Deployment: AWS G6 and G6e GPU instances
- Cloud Training: Vast.ai (H100 GPU)

<img width="502" alt="Project Overview" src="https://github.com/user-attachments/assets/43fa0d2b-2683-41e5-a6a8-52f394d76e2e" />
