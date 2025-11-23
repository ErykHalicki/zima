from .zima_dataset import ZimaDataset
from torch.utils.data import Dataset
import numpy as np
import bisect
import time

class ZimaTorchDataset(ZimaDataset, Dataset):
    def __init__(self, file_path, 
                 sample_transform=None, 
                 max_cached_episodes=5, 
                 max_cached_images=10000, 
                 action_chunk_size=1,
                 action_history_size=0,
                 image_history_size=0):
        '''
        file_path: path to hdf5 Dataset
        sample_transform: transform function that takes in a sample dict 
            e.g. tranform({actions: [action], images: [image]})
            returned sample dict will match the key names given inside the episode (e.g. images, actions)
        max_cached_episodes: number of episodes to keep in memory using LRU cache
        max_cached_images: number of transformed images to keep in cache
        action_chunk_size: number of future actions to retreive at each get_item call
        action_history_size: number of past actions to retreive at each get_item call
        '''
        super().__init__(file_path)

        self.episode_lengths = [self.get_episode_length(i) for i in range(self.num_episodes)]
        self.episode_boundaries = np.cumsum([0] + self.episode_lengths)

        self.sample_transform = sample_transform
        self.max_cached_episodes = max_cached_episodes

        self._episode_cache = {}
        self._cache_order = []

        self._transform_cache = {}
        self._transform_cache_order = []
        self.max_cached_transforms = max_cached_images
        self.action_chunk_size = action_chunk_size
        self.action_history_size = action_history_size
        self.image_history_size = image_history_size

    def __len__(self):
        return self.episode_boundaries[-1]

    def _get_cached_episode(self, episode_num):
        if episode_num in self._episode_cache:
            self._cache_order.remove(episode_num)
            self._cache_order.append(episode_num)
            #print("cache hit!")
            return self._episode_cache[episode_num]
        
        start = time.time()
        episode = self.read_episode(episode_num)
        #print(f"episode read took {time.time()-start}")

        self._episode_cache[episode_num] = episode
        self._cache_order.append(episode_num)

        if len(self._episode_cache) > self.max_cached_episodes:
            oldest = self._cache_order.pop(0)
            del self._episode_cache[oldest]

        return episode

    def __getitem__(self, idx):
        if idx in self._transform_cache:
            self._transform_cache_order.remove(idx)
            self._transform_cache_order.append(idx)
            return self._transform_cache[idx]

        episode_num = bisect.bisect_right(self.episode_boundaries, idx) - 1
        idx_in_episode = idx - self.episode_boundaries[episode_num]

        episode = self._get_cached_episode(episode_num)

        image_start_idx = idx_in_episode - self.image_history_size

        if image_start_idx < 0:
            available_history = idx_in_episode + 1
            images = episode["images"][0:idx_in_episode + 1].copy()

            history_padding_needed = self.image_history_size - available_history + 1
            if history_padding_needed > 0:
                first_image = episode["images"][0]
                image_padding = np.repeat(first_image[np.newaxis, ...], history_padding_needed, axis=0)
                images = np.concatenate([image_padding, images], axis=0)
        else:
            images = episode["images"][image_start_idx:idx_in_episode + 1].copy()

        # Get action history (past actions before idx, not including idx) with padding if needed

        history_start_idx = idx_in_episode - self.action_history_size
        if history_start_idx < 0:
            available_history = idx_in_episode
            history_actions = episode["actions"][0:idx_in_episode].copy()

            history_padding_needed = self.action_history_size - available_history
            if history_padding_needed > 0:
                action_shape = episode["actions"].shape[1:] if len(episode["actions"].shape) > 1 else (episode["actions"].shape[0],)
                history_padding = np.zeros((history_padding_needed,) + action_shape, dtype=episode["actions"].dtype)
                history_actions = np.concatenate([history_padding, history_actions], axis=0)
        else:
            history_actions = episode["actions"][history_start_idx:idx_in_episode].copy()

        # Get future actions (including idx) with padding if needed
        future_end_idx = idx_in_episode + self.action_chunk_size
        if future_end_idx > self.episode_lengths[episode_num]:
            available_future = self.episode_lengths[episode_num] - idx_in_episode
            future_actions = episode["actions"][idx_in_episode:idx_in_episode + available_future].copy()

            future_padding_needed = self.action_chunk_size - available_future
            if future_padding_needed > 0:
                action_shape = episode["actions"].shape[1:] if len(episode["actions"].shape) > 1 else (episode["actions"].shape[0],)
                future_padding = np.zeros((future_padding_needed,) + action_shape, dtype=episode["actions"].dtype)
                future_actions = np.concatenate([future_actions, future_padding], axis=0)
        else:
            future_actions = episode["actions"][idx_in_episode:future_end_idx].copy()

        sample = {"images": images,
                  "action_history": history_actions,
                  "action_chunk": future_actions}

        if self.sample_transform:
            sample = self.sample_transform(sample)

        self._transform_cache[idx] = sample
        self._transform_cache_order.append(idx)

        if len(self._transform_cache) > self.max_cached_transforms:
            oldest = self._transform_cache_order.pop(0)
            del self._transform_cache[oldest]

        return sample

