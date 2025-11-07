import h5py
import numpy as np
import os

class ZimaDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.num_episodes = 0

        if not os.path.exists(self.file_path):
            with h5py.File(self.file_path, 'w') as f:
                pass

        with h5py.File(f'{self.file_path}', 'r') as f:
            self.num_episodes = len(f.keys())

    def add_episode(self, new_episode_dict):
        #expects dictionary of list of numpy arrays
        with h5py.File(self.file_path, 'a') as f:
            while(f"episode_{self.num_episodes}" in f.keys()):
                self.num_episodes += 1
            ep = f.create_group(f'episode_{self.num_episodes}')
            for key in new_episode_dict:
                ep.create_dataset(key, data=np.array(new_episode_dict[key]), compression='gzip', compression_opts=4)
        self.num_episodes+=1
        return self.num_episodes-1

    def read_episode(self, episode_num):
        if episode_num > self.num_episodes:
            with h5py.File(self.file_path, 'r') as f:
                return f[f"episode_{episode_num}"]
        else:
            raise Exception(f"Invalid episode_num: {episode_num}. There are only {self.num_episodes} episodes in the dataset!")
