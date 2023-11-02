import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import nibabel as nib

from src.components.utils.settings import Settings


SETTINGS = Settings()

def slice(dataset_name: str, is_training: bool):
    """
    Slice the dataset based on the given slice index.

    Args:
        dataset_name (str): The name of the dataset.
        slice_index (int): The index of the slice. Must be 0, 1, or 2.
        is_training (bool): Flag indicating whether the dataset is for training.

    Raises:
        AssertionError: If the slice index is not 0, 1, or 2.

    Returns:
        None
    """

    # Get the paths for the sliced and raw data based on the dataset name
    path_train_sliced = SETTINGS.data[dataset_name]["path_train_sliced"]
    path_train_raw = SETTINGS.data[dataset_name]["path_train_raw"]
    path_test_sliced = SETTINGS.data[dataset_name]["path_test_sliced"]
    path_test_raw = SETTINGS.data[dataset_name]["path_test_raw"]
    slice_min = SETTINGS.data[dataset_name]["slice_min"]
    slice_max = SETTINGS.data[dataset_name]["slice_max"]
    slice_axis = SETTINGS.data[dataset_name]["slice_axis"]
    
    # Check if the slice index is valid
    assert slice_axis in [0, 1, 2], 'Please select slice axis 0, 1 or 2'

    # Get the brain slice paths and brain paths based on the training flag
    if is_training:
        brain_slice_paths = glob.glob(f'{path_train_sliced}/*.jpg')
        brain_paths = glob.glob(f'{path_train_raw}/*/*_t1.nii')
    else:
        brain_slice_paths = glob.glob(f'{path_test_sliced}/*.jpg')
        brain_paths = glob.glob(f'{path_test_raw}/*/*_t1.nii')

    # Remove old slices
    for slice_path in tqdm(brain_slice_paths, desc='Remove old slices'):
        os.remove(slice_path)

    # Slice the data
    for path in tqdm(brain_paths, desc=f'Slicing data | Training data: {is_training}'):
        data = nib.load(path).get_fdata()
        for k in range(slice_min, slice_max):
            if slice_axis == 0:
                clipped = data[k,:,:]
            elif slice_axis == 1:
                clipped = data[:,k,:]
            else:
                clipped = data[:,:,k]
            
            clipped *= (255.0/clipped.max())
            image_data = Image.fromarray(clipped).convert('L')
            path_image_slice = path_train_sliced if is_training else path_test_sliced
            image_data.save(f'{path_image_slice}{path.split("/")[-1].split("_")[-2]}-{k:05d}.jpg')
        
if __name__ == '__main__':
    slice(dataset_name='BraTS2020', is_training=True) # training data
    slice(dataset_name='BraTS2020', is_training=False) # Validation data
        