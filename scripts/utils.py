import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk

class AMOSDataset(Dataset):
    """AMOS dataset."""

    def __init__(self, json_path = "task1_dataset.json", root_dir = "E://Guided Research/AMOS22/", transform=None, train_size = 160, is_val = False):
        """
        Args:
            json_path (string): Path to the json file specifying (img,label) pairs and some metadata.
            root_dir (string): Directory with the dateset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            train_size (int): how many images to use. If is_val is False, it will use the first train_size images as train. If is_val is True the last train_size images are used
            is_val (bool): if this is the training or the validation dataset.
        """
        self.is_val = is_val
        json_full_path = os.path.join(root_dir,json_path)
        self.json_file = json.load(open(json_full_path,'r'))
        if is_val:
            # list of validation images
            self.training_list = self.json_file['training'][train_size:]      
        else:
            # list of training images
            self.training_list = self.json_file['training'][:train_size]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.training_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_label_pair = self.training_list[idx]

        img_name = os.path.join(self.root_dir,
                                img_label_pair['image'])
        # image = sitk.ReadImage(img_name)
        
        label_name = os.path.join(self.root_dir,
                                img_label_pair['label'])            
        # label = sitk.ReadImage(label_name)


        sample = {'image': img_name, 'label': label_name}
        print(f'before {"eval" if self.is_val else "train"} transform')
        if self.transform:
            sample = self.transform(sample)
        print(f'after {"eval" if self.is_val else "train"} transform')
        return sample
    

def resampling_method(volume, new_spacing, interpolator=sitk.sitkLinear, default_value=0):
    """
    It resamples the original volume to have the voxel size equal to the desired one.
    Parameters
    ----------
    volume: sitk image 
        The original volume
    new_spacing: numpy.array of float (i.e. [1.15, 1.30, 0.75])
        The desired voxel size
    Returns
    ----------
    sitk image
        The input volume resampled with the desired voxel size
    """
    original_size = volume.GetSize()
    original_spacing = volume.GetSpacing()
    new_size = [int((original_size[0] - 1) * original_spacing[0] / new_spacing[0]),
                int((original_size[1] - 1) *
                    original_spacing[1] / new_spacing[1]),
                int((original_size[2] - 1) * original_spacing[2] / new_spacing[2])]
    new_volume = sitk.Resample(volume, new_size, sitk.Transform(), interpolator, volume.GetOrigin(),
                               new_spacing, volume.GetDirection(), default_value, volume.GetPixelID())
    return new_volume


def create_bladder_voxels(label_img, bladder_indicator=14):
    """
    Create segmentation mask for bladder only
    Parameters
    ----------
    label_img: sitk image
        the original image
    bladder_indicator: int/float
        the number originally indicating the bladder class
        bladder indicator for amos is 14, for ctorg is is 2.0 (float)
    """
    

    label_array = sitk.GetArrayFromImage(label_img)

    bladder_voxels = np.isclose(label_array, bladder_indicator)*1.0

    bladder_voxels_img = sitk.GetImageFromArray(bladder_voxels)
    bladder_voxels_img.SetSpacing(label_img.GetSpacing())
    bladder_voxels_img.SetOrigin(label_img.GetOrigin())
    bladder_voxels_img.SetDirection(label_img.GetDirection())
    return bladder_voxels_img

