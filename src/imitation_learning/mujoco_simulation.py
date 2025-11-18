import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
from sim.web_controller import WebController
from sim.nn_controller import NNController
from datasets.zima_dataset import ZimaDataset
from models.action_resnet import ActionResNet
import sys
import torch
import threading
sys.path.append(".")#hack to use all packages in this directory

from teleop_server import start_server, get_control_state, set_camera_frame, update_mode, print_to_terminal

SPAWN_VIEWER = False

threading.Thread(target=start_server, daemon=True).start()

model = mujoco.MjModel.from_xml_path("sim/scenes/simple_scene.xml")

mjdata = mujoco.MjData(model)

# Dynamically count lights at startup
num_lights = 0
while True:
    light_name = f"light_{num_lights}"
    light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
    if light_id == -1:
        break
    num_lights += 1
print_to_terminal(f"Found {num_lights} controllable lights in the scene")

# Dynamically count rubiks cubes at startup
num_rubiks_cubes = 0
while True:
    cube_name = f"rubiks_cube{num_rubiks_cubes + 1}"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
    if body_id == -1:
        break
    num_rubiks_cubes += 1
print_to_terminal(f"Found {num_rubiks_cubes} rubiks cubes in the scene")

# Create renderer for camera capture
renderer = mujoco.Renderer(model, height=240, width=320)

# Simulation parameters
physics_fps = 2000
dt = 1.0 / physics_fps
model.opt.timestep = dt

camera_fps = 10
data_sample_fps = 10
capture_interval = 1.0 / camera_fps
data_sample_interval = 1.0 / data_sample_fps
last_capture_time = 0
last_data_time = 0

def move_item(item_name, x, y, z=0.3, randomize_orientation=False):
    """Move a named item to specified position.

    Args:
        item_name: Name of the body to move
        x, y, z: Position coordinates
        randomize_orientation: If True, randomize the z-axis rotation
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item_name)
    if body_id == -1:
        print(f"{item_name} not found!")
        return

    joint_id = model.body_jntadr[body_id]
    qpos_addr = model.jnt_qposadr[joint_id]
    dof_addr = model.jnt_dofadr[joint_id]

    mjdata.qpos[qpos_addr:qpos_addr+3] = [x, y, z]

    if randomize_orientation:
        # Random rotation around z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        # Quaternion for rotation around z-axis: [cos(θ/2), 0, 0, sin(θ/2)]
        mjdata.qpos[qpos_addr+3:qpos_addr+7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
    else:
        mjdata.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]

    mjdata.qvel[dof_addr:dof_addr+6] = 0
    mujoco.mj_forward(model, mjdata)

def randomize_lights(min_lights=0, max_lights=None):
    """Randomly toggle ceiling lights on/off using normal distribution.

    Args:
        min_lights: Minimum number of lights to keep on
        max_lights: Maximum number of lights to turn on (defaults to all lights)
    """
    if max_lights is None:
        max_lights = num_lights

    # Sample from normal distribution
    mean = (min_lights + max_lights) / 2
    std = num_lights / 4
    num_active = int(np.random.normal(mean, std))

    # Clamp to valid range
    num_active = np.clip(num_active, min_lights, max_lights)

    # Randomly select which lights to turn on
    active_indices = np.random.choice(num_lights, size=num_active, replace=False)

    # Toggle all lights
    for i in range(num_lights):
        light_name = f"light_{i}"
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
        if light_id != -1:
            model.light_active[light_id] = 1 if i in active_indices else 0

    mujoco.mj_forward(model, mjdata)

def randomize_rubiks_cubes():
    """Place one random rubiks cube in the room, rest on the desk."""
    if num_rubiks_cubes == 0:
        return

    # Pick a random cube to place in the room
    random_cube_idx = np.random.randint(0, num_rubiks_cubes)

    # Desk location (from room.xml, desk is at x=-1.15, y=0.5, desktop at z=0.2)
    desk_x = -1.15
    desk_y = 0.5
    desk_z = 0.25  # Just above desktop

    for i in range(num_rubiks_cubes):
        cube_name = f"rubiks_cube{i + 1}"

        if i == random_cube_idx:
            # Place this cube randomly in the room with random orientation
            # Room bounds: approximately -1.5 to 1.5 in x and y, avoiding furniture
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            z = 0.025  # Half the cube size (0.025) to sit on floor
            move_item(cube_name, x, y, z, randomize_orientation=True)
        else:
            # Stack cubes on desk with random orientations
            # Calculate stack position (first cube on desk, then stack upward)
            desk_cube_idx = i if i < random_cube_idx else i - 1
            stack_height = desk_cube_idx * 0.05  # Each cube is 0.05 tall
            move_item(cube_name, desk_x, desk_y, desk_z + stack_height, randomize_orientation=True)


train_mode = True
ACTION_HISTORY_SIZE = -1
ACTION_SIZE = 4
MODEL_PATH = "models/weights/action_resnet_latest.pt"

def load_model():
    """Load model and update global action parameters from model metadata."""
    global ACTION_CHUNK_SIZE, ACTION_HISTORY_SIZE, ACTION_SIZE

    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    ACTION_HISTORY_SIZE = checkpoint['metadata']['action_history_size']
    ACTION_SIZE = checkpoint['metadata']['action_size']

    return NNController(MODEL_PATH)

dataset = ZimaDataset("datasets/data/rubiks_cube_navigation_sim.hdf5")
controller = WebController(get_control_state)
action_history_buffer = []
if not train_mode:
    controller = load_model()

episode_data = {"images": [], "actions": []}
reset_episode = True
box_spawn_range = 0.75

fps_counter = 0
fps_start_time = time.time()
fps_report_interval = 3.0

print_to_terminal("Teleop server started at http://localhost:5000")
update_mode("TRAIN" if train_mode else "TEST")

if SPAWN_VIEWER:
    viewer = mujoco.viewer.launch_passive(model, mjdata, show_left_ui=False, show_right_ui=False)
else:
    viewer = None

try:
    while viewer is None or viewer.is_running():
        control_state = get_control_state()

        if reset_episode:
            robot_x = np.random.uniform(-0.3, 0.3)
            robot_y = np.random.uniform(-0.3, 0.3)
            move_item("robot_body", robot_x, robot_y, 0.01, randomize_orientation=True)
            randomize_lights(min_lights=5)
            randomize_rubiks_cubes()
            action_history_buffer.clear()
            reset_episode = False

        if control_state['toggle_mode']:
            train_mode = not train_mode
            if train_mode:
                controller = WebController(get_control_state)
                print_to_terminal("Switched to TRAIN mode")
                update_mode("TRAIN")
            else:
                controller = load_model()
                print_to_terminal("Switched to TEST mode")
                update_mode("TEST")

        step_start = time.time()

        mujoco.mj_step(model, mjdata)
        if viewer is not None:
            viewer.sync()

        if mjdata.time - last_capture_time >= capture_interval:
            fps_counter += 1
            renderer.update_scene(mjdata, camera="front_camera")
            rgb_array = renderer.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            if train_mode:
                controller.update(model, mjdata)
            else:
                if len(action_history_buffer) < ACTION_HISTORY_SIZE:
                    padding_needed = ACTION_HISTORY_SIZE - len(action_history_buffer)
                    action_history = np.zeros((padding_needed, ACTION_SIZE))
                    if len(action_history_buffer) > 0:
                        action_history = np.concatenate([action_history, np.array(action_history_buffer)])
                else:
                    action_history = np.array(action_history_buffer)
                
                controller.update(model, mjdata, rgb_array, action_history)

                executed_action = ActionResNet.bin_action(controller.get_normalized_speeds())
                
                action_history_buffer.append(executed_action)
                if len(action_history_buffer) > ACTION_HISTORY_SIZE:
                    action_history_buffer.pop(0)

            set_camera_frame(bgr_array)

            if mjdata.time - last_data_time >= data_sample_interval:
                episode_data["images"].append(bgr_array)
                episode_data["actions"].append(controller.get_normalized_speeds())
                last_data_time = mjdata.time
            last_capture_time = mjdata.time

        if control_state['save_episode']:
            save_start_time = time.time()
            episode_num = dataset.add_episode(episode_data)
            print_to_terminal(f"episode_{episode_num} saved with {len(episode_data['images'])} frames. Save time: {time.time()-save_start_time:.2f}s")
            episode_data["images"].clear()
            episode_data["actions"].clear()
            reset_episode = True

        if control_state['discard_episode']:
            print_to_terminal(f"Episode discarded ({len(episode_data['images'])} frames)")
            episode_data["images"].clear()
            episode_data["actions"].clear()
            reset_episode = True

        if time.time() - fps_start_time >= fps_report_interval:
            fps = fps_counter / (time.time() - fps_start_time)
            print_to_terminal(f"Simulation FPS: {fps:.1f}")
            fps_counter = 0
            fps_start_time = time.time()

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
finally:
    if viewer is not None:
        viewer.close()
