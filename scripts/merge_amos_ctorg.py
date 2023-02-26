import numpy as np
import socket
import os
import json
import SimpleITK as sitk
# from distutils.dir_util import copy_tree
import shutil

# on polyaxon
amos_dir = "/data/dan_blanaru/AMOS22_preprocessed/"
ctorg_dir = "/data/dan_blanaru/CTORG_preprocessed/"
target_dir = "/data/dan_blanaru/merged_AMOS_CTORG/"
# these are parameters that you should modify, since i kept them fixed i didn't pass them as arguments
######################################################################################################
os.makedirs(target_dir, exist_ok=True)
json_amos = json.load(open(amos_dir+"task1_dataset.json", 'r'))
json_ctorg = json.load(open(ctorg_dir+"task1_dataset.json", 'r'))


amos_training_list = json_amos['training']
ctorg_training_list = json_ctorg['training']

target_json = {
    "name": "merged CTORG and AMOS",
    "description": f"the first {len(amos_training_list)} images are from AMOS, the rest {len(ctorg_training_list)} are from CTORG",
    "tensorImageSize": "3D",
    "modality": {"0": "CT"},
    "labels": {"0": "background", "1": "bladder"},
    "numTraining": len(amos_training_list) + len(ctorg_training_list)
}
target_training_list = []
index = 0
for path_pair in amos_training_list:
    img_path = amos_dir+path_pair['image']
    label_path = amos_dir + path_pair['label']
    new_filename = f"merged_{index:03d}.nii.gz"

    shutil.copy(img_path, target_dir+"/imagesTr/"+new_filename)
    shutil.copy(label_path, target_dir+"/labelsTr/"+new_filename)
    target_training_list.append({"image": f"./imagesTr/{new_filename}",
                                 "label": f"./labelsTr/{new_filename}"})
    
    index = index+1

print(f"moved {len(amos_training_list)} images from AMOS to the merged folder")
for path_pair in ctorg_training_list:
    img_path = ctorg_dir+path_pair['image']
    label_path = ctorg_dir + path_pair['label']
    new_filename = f"merged_{index:03d}.nii.gz"

    shutil.copy(img_path, target_dir+"/imagesTr/"+new_filename)
    shutil.copy(label_path, target_dir+"/labelsTr/"+new_filename)
    target_training_list.append({"image": f"./imagesTr/{new_filename}",
                                 "label": f"./labelsTr/{new_filename}"})
    index = index+1

target_json['training'] = target_training_list
print(f"moved {len(ctorg_training_list)} images from AMOS to the merged folder")
json.dump(target_json,open(target_dir+"task1_dataset.json","w"))

# target_json = open(target_dir+"task1_dataset.json",'r')
# generate json for ctorg
# resize AMOS script accomodates CT_ORG
# combine them
