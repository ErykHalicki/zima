from .zima_dataset import ZimaDataset
from torch.utils.data import Dataset
import numpy as np
import bisect

class ZimaTorchDataset(ZimaDataset, Dataset):
    def __init__(self, file_path, sample_transform=None, max_cached_episodes=10):
        '''
        file_path: path to hdf5 Dataset
        sample_transform: transform function that takes in a sample dict 
            e.g. tranform({actions: [action], images: [image]})
            returned sample dict will match the key names given inside the episode (e.g. images, actions)
        max_cached_episodes: number of episodes to keep in memory using LRU cache
        '''
        super().__init__(file_path)

        self.episode_lengths = [self.get_episode_length(i) for i in range(self.num_episodes)]
        self.episode_boundaries = np.cumsum([0] + self.episode_lengths)

        self.sample_transform = sample_transform
        self.max_cached_episodes = max_cached_episodes

        self._episode_cache = {}
        self._cache_order = []

    def __len__(self):
        return self.episode_boundaries[-1]

    def _get_cached_episode(self, episode_num):
        if episode_num in self._episode_cache:
            self._cache_order.remove(episode_num)
            self._cache_order.append(episode_num)
            return self._episode_cache[episode_num]

        episode = self.read_episode(episode_num)

        self._episode_cache[episode_num] = episode
        self._cache_order.append(episode_num)

        if len(self._episode_cache) > self.max_cached_episodes:
            oldest = self._cache_order.pop(0)
            del self._episode_cache[oldest]

        return episode

    def __getitem__(self, idx):
        episode_num = bisect.bisect_right(self.episode_boundaries, idx) - 1
        idx_in_episode = idx - self.episode_boundaries[episode_num]

        episode = self._get_cached_episode(episode_num)

        sample = {key: episode[key][idx_in_episode] for key in episode}
    
        if self.sample_transform:
            sample=self.sample_transform(sample)

        return sample

