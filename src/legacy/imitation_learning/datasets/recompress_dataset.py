import h5py
import argparse
import os
import shutil
from tqdm import tqdm

def recompress_dataset(input_path, output_path, compression=None, compression_opts=None):
    """
    input_path: path to input hdf5 file
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

    with h5py.File(input_path, 'r') as f_in:
        num_episodes = len(f_in.keys())

        with h5py.File(output_path, 'w') as f_out:
            for episode_name in tqdm(f_in.keys(), desc="Recompressing episodes"):
                ep_in = f_in[episode_name]
                ep_out = f_out.create_group(episode_name)

                for key in ep_in.keys():
                    data = ep_in[key][:]
                    ep_out.create_dataset(
                        key,
                        data=data,
                        compression=compression,
                        compression_opts=compression_opts
                    )

    in_size = os.path.getsize(input_path) / (1024**2)
    out_size = os.path.getsize(output_path) / (1024**2)

    print(f"\nInput size: {in_size:.2f} MB")
    print(f"Output size: {out_size:.2f} MB")
    print(f"Compression ratio: {out_size/in_size:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompress HDF5 dataset with different compression settings")
    parser.add_argument("input", help="Input HDF5 file path")
    parser.add_argument("output", help="Output HDF5 file path")
    parser.add_argument("--compression", default=None, choices=[None, 'gzip', 'lzf', 'szip'],
                        help="Compression method (default: None)")
    parser.add_argument("--compression-level", type=int, default=None,
                        help="Compression level for gzip (1-9, default: 4)")

    args = parser.parse_args()

    recompress_dataset(
        args.input,
        args.output,
        compression=args.compression,
        compression_opts=args.compression_level
    )
