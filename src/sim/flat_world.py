import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("scenes/simple_scene.xml")

data = mujoco.MjData(model)

# Simulation parameters
physics_fps = 2000
dt = 1.0 / physics_fps
model.opt.timestep = dt

# FPS tracking
frame_count = 0
fps_update_interval = 100
last_fps_time = time.time()

print("Starting MuJoCo simulation with passive viewer...")
print(f"DOF: {model.nv}")
print(f"Actuators: {model.nu}")
print(f"Physics FPS: {physics_fps}")
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
        for i in range(min(2, model.nu)):
            data.ctrl[i] = 0.01

# Launch passive viewer - this gives us full control over the simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        #controller(model, data) #abstract this away to another file
        data.actuator('motor_l_wheel').ctrl = -1.0
        data.actuator('motor_r_wheel').ctrl = 1.0
        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Simulation ended.")
