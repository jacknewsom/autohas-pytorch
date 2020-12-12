from scipy.stats import multivariate_normal
from itertools import product
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

class DummyPose(torch.utils.data.Dataset):
    def __init__(self, length, seed=42, load_torch=True):
        self.length = length
        self.seed = seed
        self.volume = [100, 100, 100]
        # return as tensor?
        self.load_torch = load_torch

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        '''
        Create something like a muon track, but target is sparse
        heatmaps
        '''
        if i < 0 or i >= self.length:
            i %= self.length

        np.random.seed(i*self.seed)
        start, end = np.random.randint(0, self.volume, (2, 3))
        distance = np.linalg.norm(start-end).astype(int)

        locations = np.linspace(start, end, num=distance).astype(int)
        features = np.ones((len(locations), 1))

        # evaluate standard multivariate normal on grid
        densities = np.zeros(len(locations))
        for point in start, end:
            gaussian = multivariate_normal(point, [1 for i in self.volume])
            densities += gaussian.pdf(locations)
        densities /= 2

        densities = densities.reshape(-1, 1)

        if self.load_torch:
            locations = torch.from_numpy(locations).long()
            features = torch.from_numpy(features).float()
            densities = torch.from_numpy(densities).float()

        return locations, features, densities
