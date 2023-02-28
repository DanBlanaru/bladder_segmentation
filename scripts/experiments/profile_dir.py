import numpy as np
import os
import nibabel as nib

def profile_images(base_folder,img_list):
    csv_path = f"/data/dan_blanaru/preprocessed_data/CTORG/{base_folder.split('/')[-1]}.csv"
    np.set_printoptions(precision=2, suppress=True)
    f = open(csv_path, 'w')
    f.write('name,dimx,dimy,dimz,sizex,sizey,sizez,imgsizex,imgsizey,imgsizez\n')

    # for filename,filename_label in zip(tr_img_list, tr_label_list):
    for filename in img_list:
        if filename == '@eaDir'or filename == '.DS_Store':
            continue
        full_path = os.path.join(base_folder,filename)
        img = nib.load(full_path)
        shape = img.get_fdata().shape
        # vox_size = img.header['pixdim']

        vox_size = img.header.get_zooms() 

        img_size = [shape[i]*vox_size[i] for i in range(3)]
        # f.write(filename)
        # f.write(img.get_fdata().shape[0])
        f.write(f"{filename},{shape[0]},{shape[1]},{shape[2]},{vox_size[0]},{vox_size[1]},{vox_size[2]},{img_size[0]},{img_size[1]},{img_size[2]}\n")
        print(f"{filename},{shape[0]},{shape[1]},{shape[2]},{vox_size[0]},{vox_size[1]},{vox_size[2]},{img_size[0]},{img_size[1]},{img_size[2]}")
    f.close()
    print('done')



tr_img_dir = '/data/dan_blanaru/preprocessed_data/CTORG/imagesTr'
tr_img_list = os.listdir(tr_img_dir)
# print("img:\n",tr_img_list)

tr_label_dir = '/data/dan_blanaru/preprocessed_data/CTORG/labelsTr'
tr_label_list = os.listdir(tr_label_dir)
# print("labels:\n",tr_label_list)

ts_img_dir = '/data/dan_blanaru/preprocessed_data/CTORG/imagesTs'
ts_img_list = os.listdir(ts_img_dir)

luis_img_dir = '/data/dan_blanaru/original_data/CTORG_Luis/ct'
luis_img_list = os.listdir(luis_img_dir)

amos_img_dir = '/data/dan_blanaru/AMOS22/imagesTr'
amos_img_list = os.listdir(amos_img_dir)

# profile_images(tr_img_dir,tr_img_list)
# profile_images(ts_img_dir,ts_img_list)
# profile_images(luis_img_dir,luis_img_list)
profile_images(amos_img_dir,amos_img_list)

