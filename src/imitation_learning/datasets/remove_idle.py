import h5py
import os
import numpy as np
from tqdm import tqdm

def bin_action(action):
    '''
    Bins a continuous action [left_speed, right_speed] into 1 of 5 classes
    Classes: 0=stop, 1=forward, 2=backward, 3=right, 4=left
    Returns: class index (int)
    '''
    left_speed, right_speed = action

    SPEED_THRESHOLD = 0.1
    TURNING_THRESHOLD = 0.1

    avg_speed = (left_speed + right_speed) / 2
    speed_diff = left_speed - right_speed

    # Determine class
    if abs(speed_diff) > TURNING_THRESHOLD:
        if speed_diff > 0:
            return 3  # right (left wheel faster)
        else:
            return 4  # left (right wheel faster)
    else:  # Forward/backward is dominant
        if avg_speed > SPEED_THRESHOLD:
            return 1  # forward
        elif avg_speed < -SPEED_THRESHOLD:
            return 2  # backward
        else:
            return 0  # stop

def find_first_non_idle_index(actions):
    '''
    Find the index of the first non-idle (non-stop) action in the episode.
    Returns the index, or len(actions) if all actions are idle.
    '''
    for i, action in enumerate(actions):
        binned = bin_action(action)
        if binned != 1:  # Not stop
            return i
    return len(actions)  # All actions are idle

def find_last_non_idle_index(actions):
    '''
    Find the index of the last non-idle (non-stop) action in the episode.
    Returns the index, or -1 if all actions are idle.
    '''
    for i in range(len(actions) - 1, -1, -1):
        binned = bin_action(actions[i])
        if binned != 1:  # Not stop
            return i
    return -1  # All actions are idle

def remove_leading_and_trailing_idle(input_path, output_path):
    """
    Remove all leading and trailing idle (stop) actions from each episode.

    input_path: path to input hdf5 file
    output_path: path to output hdf5 file
    """

    if os.path.exists(output_path):
        response = input(f"{output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        os.remove(output_path)

    total_samples_before = 0
    total_samples_after = 0
    episodes_modified = 0
    episodes_skipped = 0  # Episodes that are all idle

    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            for episode_name in tqdm(f_in.keys(), desc="Processing episodes"):
                ep_in = f_in[episode_name]
                actions = ep_in['actions'][:]
                images = ep_in['images'][:]

                total_samples_before += len(actions)

                # Find first and last non-idle actions
                first_active_idx = find_first_non_idle_index(actions)
                last_active_idx = find_last_non_idle_index(actions)

                # Skip episode if all actions are idle
                if first_active_idx >= len(actions) or last_active_idx < 0:
                    episodes_skipped += 1
                    print(f"\nSkipping {episode_name}: all actions are idle")
                    continue

                # Trim the data (inclusive of last_active_idx)
                trimmed_actions = actions[first_active_idx:last_active_idx + 1]
                trimmed_images = images[first_active_idx:last_active_idx + 1]

                total_samples_after += len(trimmed_actions)

                if first_active_idx > 0 or last_active_idx < len(actions) - 1:
                    episodes_modified += 1

                # Create episode group and save trimmed data
                ep_out = f_out.create_group(episode_name)
                ep_out.create_dataset('actions', data=trimmed_actions)
                ep_out.create_dataset('images', data=trimmed_images)

    in_size = os.path.getsize(input_path) / (1024**2)
    out_size = os.path.getsize(output_path) / (1024**2)

    samples_removed = total_samples_before - total_samples_after

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes modified: {episodes_modified}")
    print(f"Episodes skipped (all idle): {episodes_skipped}")
    print(f"Total samples before: {total_samples_before}")
    print(f"Total samples after: {total_samples_after}")
    print(f"Samples removed: {samples_removed} ({100*samples_removed/total_samples_before:.2f}%)")
    print(f"\nInput file size: {in_size:.2f} MB")
    print(f"Output file size: {out_size:.2f} MB")
    print(f"Size reduction: {in_size - out_size:.2f} MB ({100*(in_size-out_size)/in_size:.2f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    input_path = "datasets/data/orange_cube_WILLWORK.hdf5"
    output_path = "datasets/data/orange_cube_MUSTWORK.hdf5"

    remove_leading_and_trailing_idle(input_path, output_path)
