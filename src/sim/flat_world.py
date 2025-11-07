import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
from keyboard_controller import KeyboardController

model = mujoco.MjModel.from_xml_path("scenes/simple_scene.xml")

data = mujoco.MjData(model)

# Create renderer for camera capture
renderer = mujoco.Renderer(model, height=480, width=640)

# Simulation parameters
physics_fps = 2000
dt = 1.0 / physics_fps
model.opt.timestep = dt

camera_fps = 30
capture_interval = 1.0 / camera_fps
last_capture_time = 0

controller = KeyboardController()

def drop_box(x, y, z=0.3):
    """Drop the orange box at specified position."""
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_box")
    if box_body_id == -1:
        print("Orange box not found!")
        return

    joint_id = model.body_jntadr[box_body_id]
    qpos_addr = model.jnt_qposadr[joint_id]
    dof_addr = model.jnt_dofadr[joint_id]

    data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
    data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
    data.qvel[dof_addr:dof_addr+6] = 0
    mujoco.mj_forward(model, data)

print("Starting MuJoCo simulation with passive viewer...")
print(f"DOF: {model.nv}")
print(f"Actuators: {model.nu}")
print(f"Physics FPS: {physics_fps}")
print("-" * 50)

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    drop_box(1, 0)
    while viewer.is_running():
        step_start = time.time()

        controller.update(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()

        if data.time - last_capture_time >= capture_interval:
            renderer.update_scene(data, camera="front_camera")
            rgb_array = renderer.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imshow("zima front camera", bgr_array)
            cv2.waitKey(1)
            last_capture_time = data.time

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Simulation ended.")
