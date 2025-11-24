import h5py
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.action_resnet import ActionResNet
from tqdm import tqdm

def preprocess_dataset(input_path, output_path):
    with h5py.File(input_path, 'r') as f_in:
        num_episodes = len(f_in.keys())
        print(f"Found {num_episodes} episodes in dataset")

        with h5py.File(output_path, 'w') as f_out:
            for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
                episode_key = f"episode_{episode_idx}"

                if episode_key not in f_in:
                    continue

                ep_in = f_in[episode_key]
                ep_out = f_out.create_group(episode_key)

                for key in ep_in.keys():
                    if key == "images":
                        images = ep_in[key][:]

                        processed_images = []
                        for img in images:
                            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            processed_images.append(rgb_img)

                        processed_images = np.array(processed_images)
                        ep_out.create_dataset(key, data=processed_images, compression=None)
                    else:
                        ep_out.create_dataset(key, data=ep_in[key][:], compression=None)

        print(f"\nPreprocessing complete!")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_dataset.py <input_hdf5_path> <output_hdf5_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)

    if os.path.exists(output_path):
        response = input(f"Warning: Output file '{output_path}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)

    preprocess_dataset(input_path, output_path)
