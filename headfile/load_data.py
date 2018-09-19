
# coding: utf-8

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

get_ipython().run_line_magic('matplotlib', 'inline')

def load_dataset(path,filename,dataset_type = 'single',input_name='input',output_name='output'):
    
    ## This function loads the data into a numpy array
    ##
    ## dataset_type: 'single' means that each input-output pair is stored in a single file 
    
    assert os.path.isfile(os.path.join(path,filename))
    f = h5py.File(os.path.join(path,filename))
    
    if dataset_type == 'single':        
        input_dataset = np.asarray(f[input_name],dtype=np.float32)
        output_dataset = np.asarray(f[output_name],dtype=np.float32)
        
        
    f.close()
    return input_dataset,output_dataset



class Hologram_Dataset(Dataset):
    """time-of-flight scattering media dataset."""

    def __init__(self, path, dataset_type = 'single',input_name='input',output_name='output', transform=None):
        """

        """
        self.filename_list = os.listdir(path)
        self.transform = transform
        self.path = path
        self.dataset_type = dataset_type
        self.input_name = input_name
        self.output_name = output_name

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        
        assert idx < len(self.filename_list)
        filename = self.filename_list[idx]
        
        input_dataset,output_dataset = load_dataset(self.path,filename,dataset_type = 'single',input_name=self.input_name,output_name=self.output_name)

        sample = {'input': np.reshape(input_dataset,(-1,128,128)), 'output': np.reshape(output_dataset,(-1,128,128))}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image, output_image = sample['input'], sample['output']
        return {'input': torch.from_numpy(input_image),
                'output': torch.from_numpy(output_image)}

