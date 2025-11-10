import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
from sim.keyboard_controller import KeyboardController
from sim.nn_controller import NNController
from datasets.zima_dataset import ZimaDataset
from pynput import keyboard
import sys
sys.path.append(".")#hack to use all packages in this directory

model = mujoco.MjModel.from_xml_path("sim/scenes/simple_scene.xml")

mjdata = mujoco.MjData(model)

# Create renderer for camera capture
renderer = mujoco.Renderer(model, height=480, width=640)

# Simulation parameters
physics_fps = 2000
dt = 1.0 / physics_fps
model.opt.timestep = dt

camera_fps = 30
capture_interval = 1.0 / camera_fps
last_capture_time = 0

def move_item(item_name, x, y, z=0.3):
    """Move a named item to specified position."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item_name)
    if body_id == -1:
        print(f"{item_name} not found!")
        return

    joint_id = model.body_jntadr[body_id]
    qpos_addr = model.jnt_qposadr[joint_id]
    dof_addr = model.jnt_dofadr[joint_id]

    mjdata.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
    mjdata.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
    mjdata.qvel[dof_addr:dof_addr+6] = 0
    mujoco.mj_forward(model, mjdata)

def move_item_random(item_name, min_coords, max_coords):
    """Move a named item to a random position within a 3D rectangular prism.

    Args:
        item_name: Name of the body to move
        min_coords: (x_min, y_min, z_min) tuple
        max_coords: (x_max, y_max, z_max) tuple
    """
    x = np.random.uniform(min_coords[0], max_coords[0])
    y = np.random.uniform(min_coords[1], max_coords[1])
    z = np.random.uniform(min_coords[2], max_coords[2])
    move_item(item_name, x, y, z)


train_mode = False

dataset = ZimaDataset("datasets/data/green_cube_navigation.hdf5")
controller = KeyboardController()
if not train_mode:
    controller = NNController("models/weights/action_resnet_best.pt")

episode_data = {"images": [], "actions": []}
save_episode = False
discard_episode = False
reset_episode = True
box_spawn_range = 1.25 

def _on_press(key):
    global save_episode
    global discard_episode
    try:
        if key.char == 'o' and train_mode:
            save_episode = True
        if key.char == 'p':
            discard_episode = True
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=_on_press)
listener.start()

with mujoco.viewer.launch_passive(model, mjdata, show_left_ui=False, show_right_ui=False) as viewer:
    while viewer.is_running():
        if reset_episode:
            move_item_random("orange_box", (-box_spawn_range, -box_spawn_range, 0.1), (box_spawn_range, box_spawn_range, 0.1))
            move_item_random("blue_box", (-box_spawn_range, -box_spawn_range, 0.1), (box_spawn_range, box_spawn_range, 0.1))
            move_item_random("green_box", (-box_spawn_range, -box_spawn_range, 0.1), (box_spawn_range, box_spawn_range, 0.1))
            move_item("robot_body", 0, 0, 0.068)
            reset_episode = False

        step_start = time.time()
        if train_mode:
            controller.update(model, mjdata)
        mujoco.mj_step(model, mjdata)
        viewer.sync()

        if mjdata.time - last_capture_time >= capture_interval:
            renderer.update_scene(mjdata, camera="front_camera")
            rgb_array = renderer.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            if not train_mode:
                controller.update(model,mjdata, rgb_array)

            cv2.imshow("zima front camera", bgr_array)
            cv2.waitKey(1)
            last_capture_time = mjdata.time
            #--- CAPTURE DATA ---
            episode_data["images"].append(bgr_array)
            episode_data["actions"].append(controller.get_normalized_speeds())

        if save_episode:
            save_start_time = time.time()
            episode_num = dataset.add_episode(episode_data)
            len
            print(f"episode_{episode_num} saved with {len(episode_data["images"])} frames. Save time: {time.time()-save_start_time}s")
            episode_data["images"].clear()
            episode_data["actions"].clear()
            reset_episode = True
            save_episode = False

        if discard_episode:
            episode_data["images"].clear()
            episode_data["actions"].clear()
            reset_episode = True
            discard_episode = False

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

