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