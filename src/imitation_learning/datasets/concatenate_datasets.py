import h5py
import argparse
import os
from tqdm import tqdm

def concatenate_datasets(input1_path, input2_path, output_path, compression=None, compression_opts=None):
    """
    input1_path: path to first input hdf5 file
    input2_path: path to second input hdf5 file
    output_path: path to output hdf5 file
    compression: None, 'gzip', 'lzf', or 'szip'
    compression_opts: compression level (1-9 for gzip)
    """

    if os.path.exists(output_path):
        response = input(f"{output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        os.remove(output_path)

    with h5py.File(input1_path, 'r') as f1:
        with h5py.File(input2_path, 'r') as f2:
            num_episodes_1 = len(f1.keys())
            num_episodes_2 = len(f2.keys())
            total_episodes = num_episodes_1 + num_episodes_2

            with h5py.File(output_path, 'w') as f_out:
                episode_counter = 0

                for episode_name in tqdm(f1.keys(), desc="Copying episodes from dataset 1"):
                    ep_in = f1[episode_name]
                    ep_out = f_out.create_group(f"episode_{episode_counter}")

                    for key in ep_in.keys():
                        data = ep_in[key][:]
                        ep_out.create_dataset(
                            key,
                            data=data,
                            compression=compression,
                            compression_opts=compression_opts
                        )
                    episode_counter += 1

                for episode_name in tqdm(f2.keys(), desc="Copying episodes from dataset 2"):
                    ep_in = f2[episode_name]
                    ep_out = f_out.create_group(f"episode_{episode_counter}")

                    for key in ep_in.keys():
                        data = ep_in[key][:]
                        ep_out.create_dataset(
                            key,
                            data=data,
                            compression=compression,
                            compression_opts=compression_opts
                        )
                    episode_counter += 1

    out_size = os.path.getsize(output_path) / (1024**2)

    print(f"\nDataset 1 episodes: {num_episodes_1}")
    print(f"Dataset 2 episodes: {num_episodes_2}")
    print(f"Total episodes: {total_episodes}")
    print(f"Output size: {out_size:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate two HDF5 datasets")
    parser.add_argument("input1", help="First input HDF5 file path")
    parser.add_argument("input2", help="Second input HDF5 file path")
    parser.add_argument("output", help="Output HDF5 file path")
    parser.add_argument("--compression", default=None, choices=[None, 'gzip', 'lzf', 'szip'],
                        help="Compression method (default: None)")
    parser.add_argument("--compression-level", type=int, default=None,
                        help="Compression level for gzip (1-9, default: 4)")

    args = parser.parse_args()

    concatenate_datasets(
        args.input1,
        args.input2,
        args.output,
        compression=args.compression,
        compression_opts=args.compression_level
    )
