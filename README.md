# Zima: End to End Robotic Manipulation Platform

## Project Overview

Zima is a low-cost autonomous robotic rover designed as a personal research platform for developing and testing end-to-end neural networks in robotic manipulation tasks. Mainly done to allow me to test VLA fine tuning and implementation, neural network based inverse kinematics, online RL training, and anything else that piques my interest.

<img src="https://github.com/user-attachments/assets/a8970ae2-b656-4fd7-85b0-6ae1a3be184e" width="400">

## Research Objectives

## Current Development Focus
- PI_0 VLA model adaptation
- Neural Network-based Inverse Kinematics
- Long-term memory representation using directed graph model

## Current System Architecture

### Hardware Components
- Custom Articulated robotic arm
- Tracked mobile base (hacked from Recon 6.0 Rover Toy)
- ESP32 microcontroller
- Raspberry Pi 5 as primary computer

### Software Stack
- ROS2 framework
- PyTorch for neural network development
- Custom hardware interface

## Computational Infrastructure
- Local Compute: Raspberry Pi 5 + ESP32
- Deployment: AWS G6 and G6e GPU instances
- Cloud Training: Vast.ai (H100 GPU)


