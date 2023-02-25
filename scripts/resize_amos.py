import numpy as np
from utils import resampling_method,create_bladder_voxels
import os
import json
import SimpleITK as sitk
np.set_printoptions(precision=4, suppress=True)




# raw_dir = "E://Guided Research/AMOS22/"
# processed_dir = 'E://Guided Research/AMOS22_preprocessed/'
raw_dir = "/data/dan_blanaru/AMOS22/"
processed_dir = "/data/dan_blanaru/AMOS22_preprocessed/"

#folder names for traub
raw_tr_dir = raw_dir + "imagesTr/"
processed_tr_dir = processed_dir + "imagesTr/"

raw_label_dir = raw_dir + "labelsTr/"
processed_label_dir = processed_dir + "labelsTr/"

raw_test_dir = raw_dir + "imagesTs/"

csv_path = processed_dir+'resizing_logs.csv'


# this is just for debug/analysis purposes
create_log = False
if create_log:
    csv_file = open(csv_path, 'w')
    original_header = "nr_voxels_original,bladder_voxels_original,bladder_voxels_ratio_original,original_shape"
    resized_header = "nr_voxels_resized,bladder_voxels_resized,bladder_voxels_ratio_resized,resized_shape"
    csv_file.write(f"filename,{original_header},{resized_header}\n")
    print(f"{csv_path},{'exists' if os.path.exists(csv_path) else 'didnt exist, created'}")


json_original_path = raw_dir + "task1_dataset.json"
json_target_path = processed_dir + "task1_dataset.json"

dataset_json = json.load(open(json_original_path))





train_list = dataset_json['training']
train_list_resampled = []
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
        print(filename_tuple, "WAS SKIPPED")
        continue
        #skip images with no bladder voxels
 
    sitk.WriteImage(img_resized, processed_tr_dir+filename)
    sitk.WriteImage(bladder_label_resized, processed_label_dir+filename)

    # print(img_original.GetSpacing())
    print(filename)
    print(f"{bladder_label_original.GetSize()} -> {bladder_label_resized.GetSize()}")
    resampled_size_list.append(list(bladder_label_resized.GetSize()))
    train_list_resampled.append(filename_tuple)


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

copy_json = True
if copy_json:
    dataset_json['labels'] = {
        '0': 'background',
        '1': 'bladder'
    }
    dataset_json['training'] = train_list_resampled
    dataset_json['numTraining'] = len(train_list_resampled)
    json.dump(dataset_json, open(json_target_path, 'w'))
