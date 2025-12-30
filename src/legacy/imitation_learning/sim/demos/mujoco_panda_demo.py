import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the Panda robot model
from robot_descriptions.loaders.mujoco import load_robot_description
model = load_robot_description("panda_mj_description")

# Create simulation data
data = mujoco.MjData(model)

# Simulation parameters
physics_fps = 100
dt = 1.0 / physics_fps
model.opt.timestep = dt

# FPS tracking
frame_count = 0
fps_update_interval = 100
last_fps_time = time.time()

print("Starting MuJoCo simulation with passive viewer...")
print(f"Model: Panda robot")
print(f"DOF: {model.nv}")
print(f"Actuators: {model.nu}")
print(f"Physics FPS: {physics_fps}")
print("Close the viewer window to exit.")
print("-" * 50)

# Controller function
def controller(model, data):
    """
    Simple controller that applies sinusoidal joint commands.
    """
    t = data.time
    frequency = 0.5  # Hz
    amplitude = 0.1

    if model.nu > 0:  # If there are actuators
        # Apply sinusoidal control to first few actuators
        for i in range(min(3, model.nu)):
            data.ctrl[i] = amplitude * np.sin(2 * np.pi * frequency * t + i)

# Launch passive viewer - this gives us full control over the simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Apply controller
        controller(model, data)

        # Step physics
        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()

        frame_count += 1

        # Display FPS
        if frame_count % fps_update_interval == 0:
            current_time = time.time()
            elapsed = current_time - last_fps_time
            fps = fps_update_interval / elapsed
            print(f"FPS: {fps:.1f} | Sim Time: {data.time:.2f}s | qpos[0]: {data.qpos[0]:.3f}")
            last_fps_time = current_time

        # Maintain real-time speed
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Simulation ended.")
