import numpy as np
import socket
import os
import json
import SimpleITK as sitk
# import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)

# label_dir = '/data/dan_blanaru/preprocessed_data/CTORG/labelsTr'
# label_dir = '../sample_img/'
# label_dir = '\\\\nas-vab.ifl/polyaxon/data1/dan_blanaru/preprocessed_data/CTORG/labelsTr/' #running on local machine

# csv_path = "..\sample_img\label_profile.csv"
# file = open(csv_path,'w')
# file.write('filename,nr_bladder_vox\n')


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
    # print(new_size)
    new_volume = sitk.Resample(volume, new_size, sitk.Transform(), interpolator, volume.GetOrigin(),
                               new_spacing, volume.GetDirection(), default_value, volume.GetPixelID())
    return new_volume


def create_bladder_voxels(label_img, bladder_indicator=14):
    # bladder indicator for amos is 14, for ctorg is is 2.0 (float)

    # label_img = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label_img)

    bladder_voxels = np.isclose(label_array, bladder_indicator)*1.0

    bladder_voxels_img = sitk.GetImageFromArray(bladder_voxels)
    bladder_voxels_img.SetSpacing(label_img.GetSpacing())
    bladder_voxels_img.SetOrigin(label_img.GetOrigin())
    bladder_voxels_img.SetDirection(label_img.GetDirection())
    return bladder_voxels_img


raw_dir = "/data/dan_blanaru/original_data/CTORG/"
# raw_dir = "//nas-vab.ifl/polyaxon/data1/dan_blanaru/original_data/CTORG/"
processed_dir = "/data/dan_blanaru/CTORG_test_preprocessed/"
# processed_dir = "//nas-vab.ifl/polyaxon/data1/dan_blanaru/CTORG_preprocessed/"

raw_tr_dir = raw_dir + "volumes/test/"
raw_label_dir = raw_dir + "labels/test/"

processed_tr_dir = processed_dir + "imagesTr/"
processed_label_dir = processed_dir + "labelsTr/"

csv_path = processed_dir + "resizing_logs.csv"
json_path = processed_dir + "task1_dataset.json"
create_log = True
create_json = True


if create_log:
    csv_file = open(csv_path, 'w')
    original_header = "nr_voxels_original,bladder_voxels_original,bladder_voxels_ratio_original,original_shape"
    resized_header = "nr_voxels_resized,bladder_voxels_resized,bladder_voxels_ratio_resized,resized_shape"
    csv_file.write(f"filename,{original_header},{resized_header}\n")
    print(f"{csv_path},{'exists' if os.path.exists(csv_path) else 'didnt exist, created'}")

filenames_volume = os.listdir(raw_tr_dir)
filenames_label = os.listdir(raw_label_dir)
filenames_volume.sort()
filenames_label.sort()
new_filenames = []


if filenames_volume[0] == "@eaDir":
    filenames_volume = filenames_volume[1:]
if filenames_label[0] == "@eaDir":
    filenames_label = filenames_label[1:]
print(filenames_volume)
if len(filenames_volume) != len(filenames_label):
    print("train and test not matching lengths")
    # print
    quit()

for filename_volume, filename_label in zip(filenames_volume, filenames_label):
    
    print(filename_volume, end=',', flush=True)

    img_original = sitk.ReadImage(raw_tr_dir+filename_volume)
    label_original = sitk.ReadImage(raw_label_dir+filename_label)
    print("loaded", end=',')
    bladder_label_original = create_bladder_voxels(
        label_original, bladder_indicator=2.0)

    target_spacing = (2, 2, 5)
    img_resized = resampling_method(img_original, target_spacing)
    bladder_label_resized = resampling_method(
        bladder_label_original, target_spacing, interpolator=sitk.sitkNearestNeighbor)

    if sitk.GetArrayFromImage(bladder_label_resized).sum() == 0:
        print(filename_volume, "WAS SKIPPED")
        continue
        # continue  # skip images with no bladder voxels
    print("processed", end=',')
    index_nr = "0"+filename_volume[7:10]
    new_filename = "ctorg_"+index_nr+".nii.gz"
    new_filenames.append(new_filename)
    sitk.WriteImage(img_resized, processed_tr_dir+new_filename)
    sitk.WriteImage(bladder_label_resized, processed_label_dir+new_filename)

    print(new_filename)
    print(f"{bladder_label_original.GetSize()} -> {bladder_label_resized.GetSize()}")

    if create_log:
        csv_file.write(new_filename+',')

        nr_voxels_original = np.prod(bladder_label_original.GetSize())
        nr_bladder_voxels_original = sitk.GetArrayFromImage(
            bladder_label_original).sum()
        bladder_ratio_original = nr_bladder_voxels_original/nr_voxels_original

        csv_file.write(f"{nr_voxels_original},")
        csv_file.write(f"{nr_bladder_voxels_original},")
        csv_file.write(f"{100*bladder_ratio_original},")
        original_shape = bladder_label_original.GetSize()
        csv_file.write(
            f"{original_shape[0]};{original_shape[1]};{original_shape[2]},")

        nr_voxels_resized = np.prod(bladder_label_resized.GetSize())

        nr_bladder_voxels_resized = sitk.GetArrayFromImage(
            bladder_label_resized).sum()
        bladder_ratio_resized = nr_bladder_voxels_resized/nr_voxels_resized

        csv_file.write(f"{nr_voxels_resized},")
        csv_file.write(f"{nr_bladder_voxels_resized},")
        csv_file.write(f"{100*bladder_ratio_resized},")
        resized_shape = bladder_label_resized.GetSize()
        csv_file.write(
            f"{resized_shape[0]};{resized_shape[1]};{resized_shape[2]},")
        csv_file.write("\n")

if create_json:
    # new_filenames = os.listdir(processed_tr_dir)
    training_list = [{"image": f"./imagesTr/{filename}",
                      "label": f"./labelsTr/{filename}"} for filename in new_filenames]
    dataset_json = {"name": "CTORG",
                    "num_training":len(training_list),
                    "training": training_list
                    }
    json.dump(dataset_json,open(json_path,'w'))
print("end")
