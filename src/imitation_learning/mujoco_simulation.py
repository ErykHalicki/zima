import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
from sim.keyboard_controller import KeyboardController
from sim.nn_controller import NNController
from datasets.zima_dataset import ZimaDataset
from models.action_resnet import ActionResNet
from pynput import keyboard
import sys
sys.path.append(".")#hack to use all packages in this directory

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
print(f"Found {num_lights} controllable lights in the scene")

# Dynamically count rubiks cubes at startup
num_rubiks_cubes = 0
while True:
    cube_name = f"rubiks_cube{num_rubiks_cubes + 1}"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
    if body_id == -1:
        break
    num_rubiks_cubes += 1
print(f"Found {num_rubiks_cubes} rubiks cubes in the scene")

# Create renderer for camera capture
renderer = mujoco.Renderer(model, height=480, width=640)

# Simulation parameters
physics_fps = 2000
dt = 1.0 / physics_fps
model.opt.timestep = dt

camera_fps = 30
data_sample_fps = 10
capture_interval = 1.0 / camera_fps
data_sample_interval = 1.0 / data_sample_fps
last_capture_time = 0
last_data_time = 0

def move_item_random_front_arc(item_name, distance_range=(1.5, 3.0), front_angle=120, exclude_center=30):
    """Move item to random position in front arc, always visible but requiring turning.
    
    Args:
        item_name: Name of the body to move
        distance_range: (min_distance, max_distance) from robot origin
        front_angle: Total angle in degrees of front arc (centered at 0°)
        exclude_center: Angle in degrees to exclude directly in front
    """
    # Sample distance
    distance = np.random.uniform(distance_range[0], distance_range[1])
    
    # Sample angle within front arc but excluding center
    half_arc = np.deg2rad(front_angle / 2)
    half_exclude = np.deg2rad(exclude_center / 2)
    
    # Sample from left or right side of the arc
    if np.random.random() < 0.5:
        # Left side: [half_exclude, half_arc]
        angle = np.random.uniform(half_exclude, half_arc)
    else:
        # Right side: [-half_arc, -half_exclude]
        angle = np.random.uniform(-half_arc, -half_exclude)
    
    # Convert polar to cartesian
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    z = 0.01
    
    move_item(item_name, x, y, z)

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
    desk_z = 0.22  # Just above desktop

    for i in range(num_rubiks_cubes):
        cube_name = f"rubiks_cube{i + 1}"

        if i == random_cube_idx:
            # Place this cube randomly in the room
            # Room bounds: approximately -1.5 to 1.5 in x and y, avoiding furniture
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            z = 0.025  # Half the cube size (0.025) to sit on floor
            move_item(cube_name, x, y, z)
        else:
            # Stack cubes on desk
            # Calculate stack position (first cube on desk, then stack upward)
            desk_cube_idx = i if i < random_cube_idx else i - 1
            stack_height = desk_cube_idx * 0.05  # Each cube is 0.05 tall
            move_item(cube_name, desk_x, desk_y, desk_z + stack_height)


train_mode = True
ACTION_CHUNK_SIZE = 4
ACTION_HISTORY_SIZE = 4
ACTION_SIZE = 4

dataset = ZimaDataset("datasets/data/orange_cube_PLEASEWORK.hdf5")
controller = KeyboardController()
action_history_buffer = []
if not train_mode:
    controller = NNController("models/weights/action_resnet_latest.pt", ACTION_CHUNK_SIZE, ACTION_HISTORY_SIZE, ACTION_SIZE)

episode_data = {"images": [], "actions": []}
save_episode = False
discard_episode = False
reset_episode = True
box_spawn_range = 0.75

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
            move_item("robot_body", 0, 0, 0.01)
            randomize_lights(min_lights=5)
            randomize_rubiks_cubes()
            action_history_buffer.clear()
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

            cv2.imshow("zima front camera", bgr_array)
            cv2.waitKey(1)
            if mjdata.time - last_data_time >= data_sample_interval:
                episode_data["images"].append(bgr_array)
                episode_data["actions"].append(controller.get_normalized_speeds())
                last_data_time = mjdata.time
            last_capture_time = mjdata.time

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
            pass

