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
        self.json_file = json.load(open(json_path,'r'))
        if is_val:
            # list of validation images
            self.training_list = self.json_file['training'][train_size:]      
        else:
            # list of training images
            self.training_list = self.json_file['training'][:train_size]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_label_pair = self.training_list[idx]

        img_name = os.path.join(self.root_dir,
                                'imagesTr',
                                img_label_pair['image'])
        image = sitk.ReadImage(img_name)
        
        label_name = os.path.join(self.root_dir,
                                'labelsTr',
                                img_label_pair['label'])            
        label = sitk.ReadImage(label_name)


        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample