import h5py
import os
import numpy as np
from tqdm import tqdm

def subsample_dataset(input_path, output_path, subsample_rate=2):
    """
    Subsample a dataset by keeping every Nth frame.

    input_path: path to input hdf5 file
    output_path: path to output hdf5 file
    subsample_rate: keep every Nth frame (e.g., 2 = keep every 2nd frame, effectively halving framerate)
    """

    if os.path.exists(output_path):
        response = input(f"{output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        os.remove(output_path)

    total_samples_before = 0
    total_samples_after = 0
    episodes_processed = 0

    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            for episode_name in tqdm(f_in.keys(), desc="Subsampling episodes"):
                ep_in = f_in[episode_name]
                actions = ep_in['actions'][:]
                images = ep_in['images'][:]

                total_samples_before += len(actions)

                # Subsample: keep every Nth frame
                subsampled_actions = actions[::subsample_rate]
                subsampled_images = images[::subsample_rate]

                total_samples_after += len(subsampled_actions)
                episodes_processed += 1

                # Create episode group and save subsampled data
                ep_out = f_out.create_group(episode_name)
                ep_out.create_dataset('actions', data=subsampled_actions)
                ep_out.create_dataset('images', data=subsampled_images)

    in_size = os.path.getsize(input_path) / (1024**2)
    out_size = os.path.getsize(output_path) / (1024**2)

    samples_removed = total_samples_before - total_samples_after

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Subsample rate: 1/{subsample_rate} (keeping every {subsample_rate} frame(s))")
    print(f"Episodes processed: {episodes_processed}")
    print(f"Total samples before: {total_samples_before}")
    print(f"Total samples after: {total_samples_after}")
    print(f"Samples removed: {samples_removed} ({100*samples_removed/total_samples_before:.2f}%)")
    print(f"Effective framerate reduction: {100*(1 - 1/subsample_rate):.1f}%")
    print(f"\nInput file size: {in_size:.2f} MB")
    print(f"Output file size: {out_size:.2f} MB")
    print(f"Size reduction: {in_size - out_size:.2f} MB ({100*(in_size-out_size)/in_size:.2f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    input_path = "datasets/data/compressed_no_idle.hdf5"
    output_path = "datasets/data/green_navigation_final.hdf5"
    subsample_rate = 3  

    subsample_dataset(input_path, output_path, subsample_rate)
