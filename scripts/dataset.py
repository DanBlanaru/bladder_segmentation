import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk

class AMOSDataset(Dataset):
    """AMOS dataset."""

    def __init__(self, json_path = "task1_dataset.json", root_dir = "E://Guided Research/AMOS22/", transform=None):
        """
        Args:
            json_path (string): Path to the json file specifying (img,label) pairs and some metadata.
            root_dir (string): Directory with the dateset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.json_file = json.load(open(json_path,'r'))
        self.training_list = self.json_file['training']
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