import torch
import numpy as np

class DummyData(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        n_active_sites = np.random.randint(1, 1000)
        active_sites = np.random.randint(0, 100, [n_active_sites, 3])
        active_features = np.random.rand(n_active_sites, 1)
        active = np.hstack((active_sites, active_features))

        centroid = np.random.rand(3,)*2 -1

        return active, centroid