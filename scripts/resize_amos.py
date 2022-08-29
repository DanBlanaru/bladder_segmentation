import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
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
                    int((original_size[1] - 1) * original_spacing[1] / new_spacing[1]), 
                            int((original_size[2] - 1) * original_spacing[2] / new_spacing[2])]
    # print(new_size)
    new_volume = sitk.Resample(volume, new_size, sitk.Transform(), interpolator, volume.GetOrigin(),
                                new_spacing, volume.GetDirection(), default_value, volume.GetPixelID())
    return new_volume

label_list = os.listdir(label_dir)
total_bladderless = 0
total_included = 0
for filename in label_list: 
    if filename[:7] != "labels-":
        continue
    img = nib.load(os.path.join(label_dir,filename))
    img_data = img.get_fdata()

    bladder_indicator = 2.0
    new_data = np.isclose(img_data,bladder_indicator)
    nr_bladder_vox = new_data.sum()
    
    print(os.path.join(label_dir,filename), nr_bladder_vox)
    # file.write(f"{filename},{nr_bladder_vox}\n")
    print(f"{filename},{nr_bladder_vox}")
    if nr_bladder_vox == 0:
        total_bladderless = total_bladderless+1
    else:
        total_included = total_included + 1

    new_img = nib.Nifti1Image(new_data,img.affine,img.header)
    new_path = os.path.join(label_dir,('bladder'+filename))
    if nr_bladder_vox !=0:
        nib.save(new_img,new_path)
# file.write(f'total,{total_bladderless}')
# file.close()
print("total bladderless: ", total_bladderless)
print("total_included", total_included)