import os
import glob
import time
import pickle
import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
from collections import OrderedDict

storage_path = '/home/data_storage/mimic-cxr/dataset/original_mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0'
files_path = os.path.join(storage_path, 'files')
meta_path = os.path.join(storage_path, 'mimic-cxr-2.0.0-metadata.csv')
negbio_path = os.path.join(storage_path, 'mimic-cxr-2.0.0-negbio.csv')

# Following the condition of Assignment3
meta_data = pd.read_csv(meta_path, usecols = ['dicom_id', 'subject_id', 'study_id', 'ViewPosition'])
meta_data = meta_data.loc[meta_data['ViewPosition']=='AP',:]
meta_data.reset_index(drop=True, inplace=True)
meta_dicom_id_yes = meta_data['dicom_id']
label_data = pd.read_csv(negbio_path)

# outdir <= '/home/swryu/cxr_list.pickle'
def get_whole_cxr_path_satisfying_condition(outdir):
    """
    Deliberately made the process of saving directory to facilitate using it later in case of address needed.
    """
    cxr_data_path = []
    for files_sub1 in tqdm(files_p_path): # p10, p11, ... patient-level
        files_sub1_sub = os.listdir(files_sub1)
        files_sub2_path = [os.path.join(files_sub1, temp) for temp in files_sub1_sub]
        files_sub2_index = os.path.join(files_sub1, 'index.html')
        files_sub2_path.remove(files_sub2_index)
        for files_sub3 in files_sub2_path: # p1000032, ... individual patient-level [Subject-id level]
            files_sub3_sub = os.listdir(files_sub3)
            files_sub4_path = [os.path.join(files_sub3, temp) for temp in files_sub3_sub]
            files_sub4_path.remove(glob.glob(os.path.join(files_sub3, '*.html'), recursive=True)[0])
            for files_sub5 in files_sub4_path: # s50414267, ... [Study-id level]
                jpg_path_list = glob.glob(os.path.join(files_sub5, '*.jpg'))
                base_dirname = [os.path.basename(temp) for temp in jpg_path_list]
                base_dirname = list(map(lambda x: x[:-4], base_dirname))
                for index, dicom_id in enumerate(base_dirname): # Get [.jpg] file
                    if dicom_id in np.array(meta_dicom_id_yes):
                        cxr_data_path.append(jpg_path_list[index])
                        break

    with open(outdir, 'wb') as f:
        pickle.dump(cxr_data_path, f) # 33501
    f.close()
    
    return None


# train_outdir <= '/home/swryu/train_list.pickle'
# test_outdir <= '/home/swryu/test_list.pickle'
def get_train_and_test_cxr_path_satisfying_condition(input_dir, train_outdir, test_outdir):
    """
    input_dir: The input directory path where cxr path information is saved (The saved result from 'get_whole_cxr_path_satisfying_condition')
    train_out_dir: The output directory which only contains the information of train dataset
    test_out_dir: The output directory which only contains the information of test dataset
    """
    with open(input_dir, 'rb') as f:
        data = pickle.load(f)
    f.close()

    train_path = []
    test_path = []
    study_id_from_path = list(map(lambda x: parsing(x), data))
    last_number = list(map(lambda x: train_test_split_standard(x), study_id_from_path))

    for index, last_num in enumerate(last_number):
        if last_num in ['8', '9']:
            test_path.append(data[index])
        else:
            train_path.append(data[index])

    print(len(test_path))

    with open(train_outdir, 'wb') as f:
        pickle.dump(train_path, f) # 26807
    f.close()

    with open(test_outdir, 'wb') as f:
        pickle.dump(test_path, f) # 6694
    f.close()

    return None

# Will be used at get_train_and_test_cxr_path_satisfying_condition & feature_path_to_image
def parsing(data):
    temp = data.split('/')[13]
    return temp


# Will be used at label_generation
def parsing2(data):
    temp = data.split('/')[13][1:]
    return temp


def train_test_split_standard(data):
    temp = data[-1]
    return temp


def min_max_extraction(data):
    minimum = data.min()
    maximum = data.max()
    return minimum, maximum


def post_process(data, specific_mean, specific_std):
    np_mean = np.array(specific_mean)
    np_std = np.array(specific_std)
    data -= np_mean
    data /= np_std
    return data


def downsampling(img, standard=256):
    horizontal, vertical, _ = img.shape
    horizontal = horizontal - horizontal % standard
    vertical = vertical - vertical % standard
    img = img[:horizontal, :vertical, :]
    horizontal_standard = horizontal // standard
    vertical_standard = vertical // standard
    output = img[::horizontal_standard, ::vertical_standard, :]
    assert output.shape[0] == 256 & output.shape[1] == 256, 'Not cut as 256 x 256. You need to consider again'

    return output


def img_normalizing_3channel(data, specific_mean, specific_std):
    dim0_min, dim0_max = min_max_extraction(data[:,:,0])    
    dim1_min, dim1_max = min_max_extraction(data[:,:,1])    
    dim2_min, dim2_max = min_max_extraction(data[:,:,2])    
    if dim0_max != dim0_min:
        data[:,:,0] = (data[:,:,0] - dim0_min) / (dim0_max - dim0_min)
    else:
        data[:,:,0] = data[:,:,0] - dim0_min 
    if dim1_max != dim1_min:
        data[:,:,1] = (data[:,:,1] - dim1_min) / (dim1_max - dim1_min)
    else:
        data[:,:,1] = data[:,:,1] - dim1_min
    if dim2_max != dim2_min:
        data[:,:,2] = (data[:,:,2] - dim2_min) / (dim2_max - dim2_min)
    else:
        data[:,:,2] = data[:,:,2] - dim2_min

    data = post_process(data, specific_mean, specific_std)
    return data


def feature_path_to_image(indir, out_filename):
    """
    inder: directory which contains the list of training / test sets
    out_filename: open the paths & save pixel values at the directory with file name 'out_filename'
        => 'X_train.pkl' or 'X_test.pkl'
    """
    start = time.time()

    with open(indir, 'rb') as f:
        path_list = pickle.load(f)
    f.close()

    od_saving_3d_pixels_array = OrderedDict()
    study_id = list(map(lambda x: parsing(x), path_list))

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    for index, img_path in enumerate(path_list):
        img = cv2.imread(img_path)
        img = downsampling(img, 256).astype(float)
        img = img_normalizing_3channel(img, imagenet_mean, imagenet_std)
        od_saving_3d_pixels_array[study_id[index]] = img

    current_position = os.getcwd()
    img_outdir = os.path.join(current_position, out_filename)

    with open(img_outdir, 'wb') as f:
        pickle.dump(od_saving_3d_pixels_array, f)
    f.close()

    end = time.time()
    total_min = (end-start) / 60
    print(f'Total time spent: {total_min} minutes')

    return None

def label_generation(indir, out_filename):
    """
    inder: directory which contains the list of training / test sets
    out_filename: open the paths & save pixel values at the directory with file name 'out_filename'
        => 'y_train.txt' or 'y_test.txt'
    """

    with open(indir, 'rb') as f:
        path_list = pickle.load(f)
    f.close()

    path_list = list(map(lambda x: parsing2(x), path_list))
    label = label_data.loc[np.isin(label_data['study_id'], np.array(path_list)), :].reset_index()
    label = label.fillna(0)
    label.iloc[:,3:] = label.iloc[:,3:].astype(int)
    label.replace(-1, 0, inplace=True)
    label.iloc[:,3:] = label.iloc[:,3:].astype(str)
    label['study_id'] = 's' + label['study_id'].astype(str)

    label_needed_info = label.iloc[:, 2:]
    del label

    list_for_txt = []
    for index in range(label_needed_info.shape[0]):
        line = ','.join(label_needed_info.iloc[index,:].values.tolist()) + '\n'
        list_for_txt.append(line)

    current_position = os.getcwd()
    outdir = os.path.join(current_position, out_filename)
    with open(outdir, 'w') as f:
        f.writelines(list_for_txt)
    f.close()

    return None

# if __name__ == '__main__':

    # If you want to reproduce => implement in a sequence.

    # preprocess('/home/swryu/cxr_list.pickle')
    # get_train_and_test_cxr_path_satisfying_condition('/home/swryu/cxr_list.pickle', 
    #                                                 '/home/swryu/train_list.pickle',
    #                                                 '/home/swryu/test_list.pickle')
    # feature_path_to_image('/home/swryu/train_list.pickle',
    #                      'X_train.pkl')
    # feature_path_to_image('/home/swryu/test_list.pickle',
    #                      'X_test.pkl')
    # label_generation('/home/swryu/train_list.pickle',
    #                 'y_train.txt')
    # label_generation('/home/swryu/test_list.pickle',
    #                 'y_test.txt')