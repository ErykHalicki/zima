import h5py
import numpy as np
import os

from numpy._core.numeric import array

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
        # expects dictionary of list of numpy arrays
        with h5py.File(self.file_path, 'a') as f:
            while(f"episode_{self.num_episodes}" in f.keys()):
                self.num_episodes += 1
            ep = f.create_group(f'episode_{self.num_episodes}')
            last_key = None
            for key in new_episode_dict:
                if last_key:
                    if len(new_episode_dict[key]) != len(new_episode_dict[last_key]):
                        raise Exception(f"Length of {key} doesnt match {last_key}!")
                ep.create_dataset(key, data=np.array(new_episode_dict[key]), compression=None)
                last_key = key
        self.num_episodes+=1
        return self.num_episodes-1
    
    def get_episode_length(self, episode_num):
        if episode_num < self.num_episodes:
            with h5py.File(self.file_path, 'r') as f:
                last_key = None
                episode = f[f"episode_{episode_num}"]
                for key in episode:
                    if last_key:
                        if len(episode[key]) != len(episode[last_key]):
                            raise Exception(f"episode_{episode_num}: Length of {key} doesnt match {last_key}!") 
                    last_key = key
                return len(episode[last_key])
        else:
            raise Exception(f"Invalid episode_num: {episode_num}. There are only {self.num_episodes} episodes in the dataset!")

    def read_episode(self, episode_num):
        if episode_num < self.num_episodes:
            with h5py.File(self.file_path, 'r') as f:
                episode = {}
                for key in f[f"episode_{episode_num}"]:
                    episode[key] = f[f"episode_{episode_num}"][key][:]
                return episode
        else:
            raise Exception(f"Invalid episode_num: {episode_num}. There are only {self.num_episodes} episodes in the dataset!")
