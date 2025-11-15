import h5py
import numpy as np
import matplotlib.pyplot as plt

dataset_path = "datasets/data/compressed.hdf5"

all_actions = []

with h5py.File(dataset_path, 'r') as f:
    for episode_name in f.keys():
        actions = f[episode_name]["actions"][:]
        all_actions.append(actions)

all_actions = np.concatenate(all_actions, axis=0)

left_speeds = all_actions[:, 0]
right_speeds = all_actions[:, 1]

unique, counts = np.unique(all_actions, axis=0, return_counts=True)
for action, count in zip(unique, counts):
    print(f"{action}: {count} ({100*count/len(all_actions):.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(left_speeds, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Left Speed')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Left Speed Distribution')
axes[0].grid(True, alpha=0.3)

axes[1].hist(right_speeds, bins=50, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Right Speed')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Right Speed Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Total actions: {len(all_actions)}")
print(f"Left speed - min: {left_speeds.min():.3f}, max: {left_speeds.max():.3f}, mean: {left_speeds.mean():.3f}, std: {left_speeds.std():.3f}")
print(f"Right speed - min: {right_speeds.min():.3f}, max: {right_speeds.max():.3f}, mean: {right_speeds.mean():.3f}, std: {right_speeds.std():.3f}")
