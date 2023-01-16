import os
import numpy as np
import pandas as pd
import cv2

"""
pp and ap file name style: 01_VDN (02_VDN Left 2 ap image has problem)
pp: pressure
ap: anteroposterior shear stress
"""

def readPP_AP(pp_ap_data_path, ppMax=1100000.0, ppMin=0.0, apMax=180000.0, apMin=0.0):
    pp_ap_files = os.listdir(pp_ap_data_path)
    pp_ap_files.sort()
    pp_dataset = []
    ap_dataset = []
    pp_ap_label = []
    mask_dataset = []
    threshold = 0.05
    padding_row = (0, 110)
    padding_column = (0, 57)

    for file in pp_ap_files:
        path = os.path.join(pp_ap_data_path, file)
        if os.path.isdir(path):
            left = os.path.join(path, 'Left')
            right = os.path.join(path, 'Right')
            left_files = os.listdir(left)
            right_files = os.listdir(right)
            left_files.sort()
            right_files.sort()
            if '.DS' in left_files[0]:
                left_files.pop(0)
            if '.DS' in right_files[0]:
                right_files.pop(0)
            for left_file in left_files:
                left_path = os.path.join(left, left_file)
                # The file format is Left->1->right_Composite_Peak_pp.dat and Right->1->left_Composite_Peak_pp.dat
                pp_path = os.path.join(left_path, 'right_Composite-Peak_pp.dat')
                ap_path = os.path.join(left_path, 'right_Composite-Peak_ap.dat')
                if file + left_file == '13_VHC1':
                    pp_path = os.path.join(left_path, 'left_Composite-Peak_pp.dat')
                    ap_path = os.path.join(left_path, 'left_Composite-Peak_ap.dat')
                # print(file + left_file)
                if (not file + left_file == '02_VDN2') and (not file + left_file == '02_VDN3'):
                    pp_data = np.loadtxt(pp_path)
                    ap_data = np.loadtxt(ap_path)
                    pp_data = np.abs(pp_data)
                    ap_data = np.abs(ap_data)
                    pp_data = np.array((pp_data - ppMin) / (ppMax - ppMin), dtype='f')
                    ap_data = np.array((ap_data - apMin) / (apMax - apMin), dtype='f')
                    pp_data = paddingMatrix(pp_data, padding_row, padding_column)
                    ap_data = paddingMatrix(ap_data, padding_row, padding_column)
                    mask = pp_data > threshold
                    # mask[mask > threshold] = 1
                    # mask[mask <= threshold] = 0

                    label = np.append('0' + file, 'Left' + left_file)
                    pp_dataset.append(pp_data)
                    ap_dataset.append(ap_data)
                    mask_dataset.append(mask)
                    pp_ap_label.append(label)
                    # print(file + " Left" + left_file + ' done!')

            for right_file in right_files:
                right_path = os.path.join(right, right_file)
                # The file format is Left->1->right_Composite_Peak_pp.dat and Right->1->left_Composite_Peak_pp.dat
                pp_path = os.path.join(right_path, 'left_Composite-Peak_pp.dat')
                ap_path = os.path.join(right_path, 'left_Composite-Peak_ap.dat')
                pp_data = np.loadtxt(pp_path)
                ap_data = np.loadtxt(ap_path)
                pp_data = np.abs(pp_data)
                ap_data = np.abs(ap_data)
                pp_data = np.array((pp_data - ppMin) / (ppMax - ppMin), dtype='f')
                ap_data = np.array((ap_data - apMin) / (apMax - apMin), dtype='f')
                pp_data = paddingMatrix(pp_data, padding_row, padding_column)
                ap_data = paddingMatrix(ap_data, padding_row, padding_column)
                mask = pp_data > threshold

                # mask[mask > threshold] = 1
                # mask[mask <= threshold] = 0

                label = np.append('0' + file, ' Right' + right_file)
                pp_dataset.append(pp_data)
                ap_dataset.append(ap_data)
                mask_dataset.append(mask)
                pp_ap_label.append(label)
                # print(file + " Right" + right_file + ' done!')
    # print(pp_ap_label)
    # print(pp_dataset[0])
    # print(ap_dataset[0])
    # print(pp_ap_label[0][0] + '\t' + pp_ap_label[0][1])

    return pp_ap_label, pp_dataset, ap_dataset, mask_dataset


def readTem(tem_path, pp_dataset, pp_ap_label, Min=5.0, Max=45.0):
    """ temperature file name style: 001VDN """
    tem_files = os.listdir(tem_path)
    tem_files.sort()
    tem_dataset = []
    tem_label = []
    padding_row = (0, 192)
    padding_column = (0, 32)

    for file in tem_files:
        path = os.path.join(tem_path, file)
        if os.path.isdir(path) and len(file) == 6:
            """ read all the temperature images """
            # for i in range(1, 5):
            #     csv_path = os.path.join(path, str(i) + '.csv')
            #     print(csv_path)
            #
            #     raw_data = pd.read_csv(csv_path, header=None, usecols=range(1, 641), skiprows=2).values
            #     label = np.append(file, i)
            #     tem_dataset.append(raw_data)
            #     tem_label.append(label)
            """ only read 3.csv temperature images """
            i = 3
            csv_path = os.path.join(path, str(i) + '.csv')
            # print(csv_path)

            raw_data = pd.read_csv(csv_path, header=None, usecols=range(1, 641), skiprows=2).values
            label = np.append(file[:3] + '_' + file[3:], i)
            tem_dataset.append(raw_data)
            tem_label.append(label)

    # print(tem_dataset[0].shape)
    # print(tem_label)

    # t_files = [file[:3] + '_' + file[3:] for file in tem_files]
    # for i in t_files:
    #     print(i)

    """ 
    normalize and split left && right foot for each temperature image
    [0]: left; [1]: right 
    """
    tem_left_right_dataset = []
    for tem_data in tem_dataset:
        normalize_data = np.array((tem_data - Min) / (Max - Min), dtype='f')
        # normalize_data = np.array(tem_data)
        rotated_data = np.array(list(zip(*reversed(normalize_data))))
        new_array = np.split(rotated_data, [320])
        new_array = np.array([paddingMatrix(i, padding_row, padding_column) for i in new_array])
        tem_left_right_dataset.append(new_array)

    i, j = 0, 0
    train_tem = []
    label = []
    # print(len(pp_ap_label))
    # print(len(pp_ap_label[1]))
    while i < len(pp_dataset) and j < len(tem_left_right_dataset):
        if pp_ap_label[i][0] == tem_label[j][0]:
            if 'Left' in pp_ap_label[i][1]:
                train_tem.append(tem_left_right_dataset[j][0])
            elif 'Right' in pp_ap_label[i][1]:
                train_tem.append(tem_left_right_dataset[j][1])
            label.append(pp_ap_label[i][0][-3:])
            i = i + 1
        else:
            j = j + 1
            # print(str(i) + '\t' + str(j))
    return train_tem, label


def image2numpy(path, foot, i):
    image_path = os.path.join(path, foot + str(i) + '.jpg')
    img = cv2.imread(image_path, 0)
    matrix = np.asarray(img / 255)
    return matrix

def readDeltaTem(tem_path, pp_dataset, pp_ap_label):
    """ temperature file name style: 001VDN """
    tem_files = os.listdir(tem_path)
    tem_files.sort()
    left_dataset = []
    right_dataset = []
    tem_label = []
    padding_row = (0, 192)
    padding_column = (0, 32)

    for file in tem_files:
        path = os.path.join(tem_path, file)
        if os.path.isdir(path) and len(file) == 6:
            """ read all the temperature images """
            # for i in range(1, 5):
            #     csv_path = os.path.join(path, str(i) + '.csv')
            #     print(csv_path)
            #
            #     raw_data = pd.read_csv(csv_path, header=None, usecols=range(1, 641), skiprows=2).values
            #     label = np.append(file, i)
            #     tem_dataset.append(raw_data)
            #     tem_label.append(label)
            """ read 1.jpg and 3.jpg temperature images """
            i = 3
            left_original = image2numpy(path, 'L', 1)
            right_original = image2numpy(path, 'R', 1)
            left = image2numpy(path, 'L', i)
            right = image2numpy(path, 'R', i)
            delta_left = np.subtract(left, left_original)
            delta_right = np.subtract(right, right_original)

            # delta_left = paddingMatrix(delta_left, padding_row, padding_column)
            # delta_right = paddingMatrix(delta_right, padding_row, padding_column)
            delta_left = paddingMatrix(left, padding_row, padding_column)
            delta_right = paddingMatrix(right, padding_row, padding_column)

            label = np.append(file[:3] + '_' + file[3:], i)
            left_dataset.append(delta_left)
            right_dataset.append(delta_right)
            tem_label.append(label)

    # print(tem_dataset[0].shape)
    # print(tem_label)

    # t_files = [file[:3] + '_' + file[3:] for file in tem_files]
    # for i in t_files:
    #     print(i)

    """ 
    normalize and split left && right foot for each temperature image
    [0]: left; [1]: right 
    """
    # tem_left_right_dataset = []
    # for tem_data in tem_dataset:
    #     normalize_data = np.array((tem_data - Min) / (Max - Min), dtype='f')
    #     # normalize_data = np.array(tem_data)
    #     rotated_data = np.array(list(zip(*reversed(normalize_data))))
    #     new_array = np.split(rotated_data, [320])
    #     new_array = np.array([paddingMatrix(i, padding_row, padding_column) for i in new_array])
    #     tem_left_right_dataset.append(new_array)

    i, j = 0, 0
    train_tem = []
    label = []
    # print(len(pp_ap_label))
    # print(len(pp_ap_label[1]))
    while i < len(pp_dataset) and j < len(left_dataset):
        if pp_ap_label[i][0] == tem_label[j][0]:
            if 'Left' in pp_ap_label[i][1]:
                train_tem.append(left_dataset[j])
            elif 'Right' in pp_ap_label[i][1]:
                train_tem.append(right_dataset[j])
            label.append(pp_ap_label[i][0][-3:])
            i = i + 1
        else:
            j = j + 1
            # print(str(i) + '\t' + str(j))
    return train_tem, label


def paddingMatrix(m, row, column):
    padding = np.pad(m, [row, column], mode='constant', constant_values=0)
    return padding