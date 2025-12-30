import h5py
from ..models.action_resnet import ActionResNet

with h5py.File("datasets/data/orange_cube_WILLWORK.hdf5", "r") as f:
    for ep_name in list(f.keys())[:10]:  # first 10 episodes
        actions = f[ep_name]["actions"][:]
        discretized = [ActionResNet.bin_action(a).argmax().item() for a in actions]
        
        print(f"{ep_name}: {discretized}")
