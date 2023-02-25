import numpy as np
from utils import resampling_method, create_bladder_voxels
import os
import json
import SimpleITK as sitk
np.set_printoptions(precision=4, suppress=True)



raw_dir = "data/dan_blanaru/original_data/CTORG/"
processed_dir = "data/dan_blanaru/CTORG_preprocessed/"

raw_tr_dir = raw_dir + "2_images_inhouse/"
raw_label_dir = raw_dir + "2_labels_inhouse/"

processed_tr_dir = processed_dir + "2_images_inhouse/"
processed_label_dir = processed_dir + "2_labels_inhouse/"

# csv_path = processed_dir + "resizing_logs.csv"
# json_path = processed_dir + "task1_dataset.json"
create_log = False
create_json = False


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
    

    img_original = sitk.ReadImage(raw_tr_dir+filename_volume)
    label_original = sitk.ReadImage(raw_label_dir+filename_label)
    print("loaded", end=',')
    bladder_label_original = create_bladder_voxels(
        label_original, bladder_indicator=1.0)

    target_spacing = (2, 2, 5)
    img_resized = resampling_method(img_original, target_spacing)
    bladder_label_resized = resampling_method(
        bladder_label_original, target_spacing, interpolator=sitk.sitkNearestNeighbor)

    if sitk.GetArrayFromImage(bladder_label_resized).sum() == 0:
        print(filename_volume, "WAS SKIPPED")
        continue
        # continue  # skip images with no bladder voxels
    print("processed", end=',')
    index_nr = filename_volume.split('.')[0]
    new_filename = index_nr+".nii.gz"
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
