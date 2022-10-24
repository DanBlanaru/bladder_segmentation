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

    ...


if socket.gethostname() == 'DESKTOP-HROVR50':
    # on local computer
    raw_dir = "E://Guided Research/AMOS22/"
    processed_dir = 'E://Guided Research/AMOS22_preprocessed/'

else:
    # on polyaxon
    raw_dir = "/data/dan_blanaru/AMOS22/"
    processed_dir = "/data/dan_blanaru/AMOS22_preprocessed/"

raw_tr_dir = raw_dir + "imagesTr/"
processed_tr_dir = processed_dir + "imagesTr/"

raw_label_dir = raw_dir + "labelsTr/"
processed_label_dir = processed_dir + "labelsTr/"

raw_test_dir = raw_dir + "imagesTs/"

csv_path = processed_dir+'resizing_logs.csv'


create_log = True
if create_log:
    csv_file = open(csv_path, 'w')
    original_header = "nr_voxels_original,bladder_voxels_original,bladder_voxels_ratio_original,original_shape"
    resized_header = "nr_voxels_resized,bladder_voxels_resized,bladder_voxels_ratio_resized,resized_shape"
    csv_file.write(f"filename,{original_header},{resized_header}\n")
    print(f"{csv_path},{'exists' if os.path.exists(csv_path) else 'didnt exist, created'}")


json_original_path = raw_dir + "task1_dataset.json"
json_target_path = processed_dir + "task1_dataset.json"

dataset_json = json.load(open(json_original_path))


copy_json = True
if copy_json:
    dataset_json['labels'] = {
        '0': 'background',
        '1': 'bladder'
    }
    json.dump(dataset_json, open(json_target_path, 'w'))


train_list = dataset_json['training']
resampled_size_list = []
for filename_tuple in train_list:
    filename = filename_tuple['image'].split('/')[-1]

    img_original = sitk.ReadImage(raw_tr_dir+filename)
    label_original = sitk.ReadImage(raw_label_dir+filename)

    bladder_label_original = create_bladder_voxels(label_original)

    target_spacing = (2, 2, 5)
    img_resized = resampling_method(img_original, target_spacing)
    bladder_label_resized = resampling_method(
        bladder_label_original, target_spacing, interpolator=sitk.sitkNearestNeighbor)

    if sitk.GetArrayFromImage(bladder_label_resized).sum() == 0:
        continue  # skip images with no bladder voxels

    sitk.WriteImage(img_resized, processed_tr_dir+filename)
    sitk.WriteImage(bladder_label_resized, processed_label_dir+filename)

    # print(img_original.GetSpacing())
    print(filename)
    print(f"{bladder_label_original.GetSize()} -> {bladder_label_resized.GetSize()}")
    resampled_size_list.append(list(bladder_label_resized.GetSize()))
    bladder_voxels_resized = sitk.GetArrayFromImage(
        bladder_label_resized)
    print(np.unique(bladder_voxels_resized,return_counts=True))


    if create_log:
        csv_file.write(filename+',')

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


print()
print(np.min(np.array(resampled_size_list), axis=0))


# todo:
# load img and label side by side

# count bladder pix in img
# resize img
# resize label
# count bladder pix in label
# if it's working, clean chrome tabs

# crop img
# crop label
# count bladder pix in label


# label_list = os.listdir(label_dir)
# total_bladderless = 0
# total_included = 0
# for filename in label_list:
#     if filename[:7] != "labels-":
#         continue
#     img = nib.load(os.path.join(label_dir,filename))
#     img_data = img.get_fdata()

#     bladder_indicator = 2.0
#     new_data = np.isclose(img_data,bladder_indicator)
#     nr_bladder_vox = new_data.sum()

#     print(os.path.join(label_dir,filename), nr_bladder_vox)
#     # file.write(f"{filename},{nr_bladder_vox}\n")
#     print(f"{filename},{nr_bladder_vox}")
#     if nr_bladder_vox == 0:
#         total_bladderless = total_bladderless+1
#     else:
#         total_included = total_included + 1

#     new_img = nib.Nifti1Image(new_data,img.affine,img.header)
#     new_path = os.path.join(label_dir,('bladder'+filename))
#     if nr_bladder_vox !=0:
#         nib.save(new_img,new_path)
# # file.write(f'total,{total_bladderless}')
# # file.close()
# print("total bladderless: ", total_bladderless)
# print("total_included", total_included)
