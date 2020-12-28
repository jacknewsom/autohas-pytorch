from scipy.stats import multivariate_normal
from itertools import product
import os
import torch
import h5py as five
import numpy as np

class MuonTracks(torch.utils.data.Dataset):

    MAX_SIZE = 1000 # too many points causes instability in sparse networks ?

    def __init__(self, data_dir):
        '''
        args:
            - data_dir (string): source directory for HDF5 files
        '''
        super(MuonTracks, self).__init__()
        if not os.path.isdir(data_dir):
            raise OSError("Directory %s does not exist" % data_dir)
        if data_dir[-1] != '/':
            data_dir += '/'
        self.data_dir = data_dir
        self.files = [self.data_dir+f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        self.default_keys = ['energy_coordinates', 'energy_values', 'sipm_coordinates', 'sipm_values', 'vertex']
        self.eps = 1e-8

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        '''
        args:
            - i (int): `i`th item from dataset to load
            - keys (iterable): (optionally,) which keys to load from `i`th item

        return:
            - sample (dict): with values..
                - sample['energy_coordinates'] (np.ndarray): sparse representation of energy depositions
                - sample['energy_values'] (np.ndarray): sparse features
                - sample['sipm_coordinates'] (np.ndarray): sparse representation of sipm readouts
                - sample['sipm_values'] (np.ndarray): sparse features
                - sample['centroid'] (np.ndarray): (3,) coordinates of arithmetic mean of
                                                   voxelized charges
        '''

        if type(i) == int:
            keys = self.default_keys
        else:
            i, keys = i

        with five.File(self.files[i], 'r') as f:
            sample = {k: f[k][:] for k in keys}

            # calculate true charge centroid
            centroid = np.mean(f['energy_coordinates'][:, :3], axis=0)
            sample['centroid'] = centroid
            vertex = f['vertex'][:]
            target = np.sqrt(np.sum(np.square(vertex-centroid)))
            sample['target'] = target

            if self.MAX_SIZE:
                for key in keys:
                    if len(sample[key]) < self.MAX_SIZE:
                        continue
                    sample[key] = sample[key][::int(len(sample[key])/self.MAX_SIZE)]
        return sample

class MuonPose(torch.utils.data.Dataset):
    '''
    Similar to MuonTracks, but for predicting heatmaps instead of points directly
    '''

    def __init__(self, data_dir,  output_shape=[128, 128, 128], dense_target=False, return_energy=False):
        super(MuonPose, self).__init__()
        if not os.path.isdir(data_dir):
            raise OSError("Directory %s does not exist" % data_dir)
        if data_dir[-1] != '/':
            data_dir += '/'
        self.data_dir = data_dir
        self.files = [self.data_dir+f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        self.default_keys = ['energy_coordinates', 'energy_values', 'sipm_coordinates', 'sipm_values',]
        self.output_shape = output_shape
        self.dense_target = dense_target
        self.return_energy = return_energy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        '''
        args:
            - i (int): `i`th item from dataset to load

        return:
        '''
        assert type(i) == int, f"Argument `i` must be an int, but {type(i)} was provided"

        with five.File(self.files[i], 'r') as f:
            np.random.seed(i)
            sample = {k: f[k][:] for k in self.default_keys}

            locations, features = sample['sipm_coordinates'], sample['sipm_values'].reshape(-1, 1)
            energy_coordinates, energy_values = sample['energy_coordinates'], sample['energy_values'].reshape(-1, 1)

            # rescale to be in region (128, 128, 128)
            start, end = energy_coordinates[0], energy_coordinates[-1]

            if not self.dense_target:
                def cartesian_product(*arrays):
                    la = len(arrays)
                    dtype = np.result_type(*arrays)
                    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
                    for i, a in enumerate(np.ix_(*arrays)):
                        arr[...,i] = a
                    return arr.reshape(-1, la)
                densities = np.zeros(len(energy_coordinates))

                mvn1 = multivariate_normal(start, [1,1,1])
                mvn2 = multivariate_normal(end, [1,1,1])

                densities += mvn1.pdf(energy_coordinates)
                densities += mvn2.pdf(energy_coordinates)
                densities /= densities.sum()
                densities = densities.reshape(-1, 1)

                if self.return_energy:
                    return ((locations, features), (energy_coordinates, energy_values)), (energy_coordinates, densities)
                return (locations, features), (heatmap, densities)
            else:
                mvn1 = multivariate_normal(start, [1,1,1])
                mvn2 = multivariate_normal(end, [1,1,1])
                target = np.zeros([128, 128, 128])
                for i in range(128):
                    for j in range(128):
                        for k in range(128):
                            x = np.array([i, j, k])
                            target[i, j, k] += mvn1.pdf(x-start) + mvn2.pdf(x-end)
                target /= 2

                if self.return_energy:
                    return ((locations, features), (energy_coordinates, energy_values)), target
                return (locations, features), target

class MuonPoseLoader:
    def __init__(self, data_dir, batch_size, device, output_shape=[128, 128, 128], dense_target=False, return_energy=False):
        self.dataset = MuonPose(data_dir, output_shape, dense_target, return_energy)
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, i):
        if self.dataset.return_energy:
            light_batch = [[], []]
            energy_batch = [[], []]
            target_batch = [[], []]

        for batch_idx in range(self.batch_size):
            data = self.dataset[i*self.batch_size+batch_idx]
            if self.dataset.return_energy:
                (light, energy), target = data
                light = [torch.from_numpy(l) for l in light]
                energy = [torch.from_numpy(e) for e in energy]
                target = [torch.from_numpy(t) for t in target]
                
                light_coordinates = torch.hstack((light[0], batch_idx * torch.ones((len(light[0]), 1))))
                energy_coordinates = torch.hstack((energy[0], batch_idx * torch.ones((len(energy[0]), 1))))
                target_coordinates = torch.hstack((target[0], batch_idx * torch.ones((len(target[0]), 1))))

                light_batch[0].append(light_coordinates)
                light_batch[1].append(light[1])

                energy_batch[0].append(energy_coordinates)
                energy_batch[1].append(energy[1])

                target_batch[0].append(target_coordinates)
                target_batch[1].append(target[1])
        if self.dataset.return_energy:
            light_batch = [torch.vstack(q).to(self.device) for q in light_batch]
            energy_batch = [torch.vstack(q).to(self.device) for q in energy_batch]
            target_batch = [q.to(self.device) for q in target_batch]
            return light_batch, energy_batch, target_batch


