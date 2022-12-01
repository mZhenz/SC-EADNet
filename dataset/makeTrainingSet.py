import os
import cv2
import random
import numpy as np
from scipy.io import loadmat, savemat
import math


def getTrainimage(raw_image, train_index, train_dir):
    i = 0
    for idx in train_index:
        # [3,3] [9,9] [15,15] [31,31]
        # [3,3] [7,7] [11,11] [15,15] [19,19] [23,23] [27,27] [31,31]
        # 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
        # pic1 = raw_image[idx[0] - 1 if idx[0] - 1 > 0 else 0:idx[0] + 2, idx[1] - 1 if idx[1] - 1 > 0 else 0:idx[1] + 2]
        # pic2 = raw_image[idx[0] - 4 if idx[0] - 4 > 0 else 0:idx[0] + 5, idx[1] - 4 if idx[1] - 4 > 0 else 0:idx[1] + 5]
        # pic3 = raw_image[idx[0] - 7 if idx[0] - 7 > 0 else 0:idx[0] + 8, idx[1] - 7 if idx[1] - 7 > 0 else 0:idx[1] + 8]
        # pic4 = raw_image[idx[0] - 15 if idx[0] - 15 > 0 else 0:idx[0] + 16, idx[1] - 15 if idx[1] - 15 > 0 else 0:idx[1] + 16]
        pic1 = raw_image[idx[0] - 1 if idx[0] - 1 > 0 else 0:idx[0] + 2, idx[1] - 1 if idx[1] - 1 > 0 else 0:idx[1] + 2]
        pic2 = raw_image[idx[0] - 2 if idx[0] - 2 > 0 else 0:idx[0] + 3, idx[1] - 2 if idx[1] - 2 > 0 else 0:idx[1] + 3]
        pic3 = raw_image[idx[0] - 3 if idx[0] - 3 > 0 else 0:idx[0] + 4, idx[1] - 3 if idx[1] - 3 > 0 else 0:idx[1] + 4]
        pic4 = raw_image[idx[0] - 4 if idx[0] - 4 > 0 else 0:idx[0] + 5, idx[1] - 4 if idx[1] - 4 > 0 else 0:idx[1] + 5]
        pic5 = raw_image[idx[0] - 5 if idx[0] - 5 > 0 else 0:idx[0] + 6, idx[1] - 5 if idx[1] - 5 > 0 else 0:idx[1] + 6]
        pic6 = raw_image[idx[0] - 6 if idx[0] - 6 > 0 else 0:idx[0] + 7, idx[1] - 6 if idx[1] - 6 > 0 else 0:idx[1] + 7]
        pic7 = raw_image[idx[0] - 7 if idx[0] - 7 > 0 else 0:idx[0] + 8, idx[1] - 7 if idx[1] - 7 > 0 else 0:idx[1] + 8]
        pic8 = raw_image[idx[0] - 8 if idx[0] - 8 > 0 else 0:idx[0] + 9, idx[1] - 8 if idx[1] - 8 > 0 else 0:idx[1] + 9]
        pic9 = raw_image[idx[0] - 9 if idx[0] - 9 > 0 else 0:idx[0] + 10, idx[1] - 9 if idx[1] - 9 > 0 else 0:idx[1] + 10]
        pic10 = raw_image[idx[0] - 10 if idx[0] - 10 > 0 else 0:idx[0] + 11, idx[1] - 10 if idx[1] - 10 > 0 else 0:idx[1] + 11]
        pic11 = raw_image[idx[0] - 11 if idx[0] - 11 > 0 else 0:idx[0] + 12, idx[1] - 11 if idx[1] - 11 > 0 else 0:idx[1] + 12]
        pic12 = raw_image[idx[0] - 12 if idx[0] - 12 > 0 else 0:idx[0] + 13, idx[1] - 12 if idx[1] - 12 > 0 else 0:idx[1] + 13]
        pic13 = raw_image[idx[0] - 13 if idx[0] - 13 > 0 else 0:idx[0] + 14, idx[1] - 13 if idx[1] - 13 > 0 else 0:idx[1] + 14]
        pic14 = raw_image[idx[0] - 14 if idx[0] - 14 > 0 else 0:idx[0] + 15, idx[1] - 14 if idx[1] - 14 > 0 else 0:idx[1] + 15]
        pic15 = raw_image[idx[0] - 15 if idx[0] - 15 > 0 else 0:idx[0] + 16, idx[1] - 15 if idx[1] - 15 > 0 else 0:idx[1] + 16]
        pic1_resize = cv2.resize(pic1, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic2_resize = cv2.resize(pic2, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic3_resize = cv2.resize(pic3, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic4_resize = cv2.resize(pic4, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic5_resize = cv2.resize(pic5, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic6_resize = cv2.resize(pic6, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic7_resize = cv2.resize(pic7, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic8_resize = cv2.resize(pic8, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic9_resize = cv2.resize(pic9, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic10_resize = cv2.resize(pic10, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic11_resize = cv2.resize(pic11, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic12_resize = cv2.resize(pic12, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic13_resize = cv2.resize(pic13, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic14_resize = cv2.resize(pic14, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic15_resize = cv2.resize(pic15, (31, 31), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_00.png'), pic1_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_01.png'), pic2_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_02.png'), pic3_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_03.png'), pic4_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_04.png'), pic5_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_05.png'), pic6_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_06.png'), pic7_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_07.png'), pic8_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_08.png'), pic9_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_09.png'), pic10_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_10.png'), pic11_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_11.png'), pic12_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_12.png'), pic13_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_13.png'), pic14_resize)
        cv2.imwrite(os.path.join(train_dir, str(i).zfill(5) + '_14.png'), pic15_resize)
        i += 1


def getValimage(raw_image, val_index, val_dir):
    i = 0
    for idx in val_index:
        # [3,3] [9,9] [15,15] [31,31]
        # pic1 = raw_image[idx[0] - 1 if idx[0] - 1 > 0 else 0:idx[0] + 2, idx[1] - 1 if idx[1] - 1 > 0 else 0:idx[1] + 2]
        # pic2 = raw_image[idx[0] - 4 if idx[0] - 4 > 0 else 0:idx[0] + 5, idx[1] - 4 if idx[1] - 4 > 0 else 0:idx[1] + 5]
        # pic3 = raw_image[idx[0] - 7 if idx[0] - 7 > 0 else 0:idx[0] + 8, idx[1] - 7 if idx[1] - 7 > 0 else 0:idx[1] + 8]
        # pic4 = raw_image[idx[0] - 15 if idx[0] - 15 > 0 else 0:idx[0] + 16, idx[1] - 15 if idx[1] - 15 > 0 else 0:idx[1] + 16]
        # pic1_resize = cv2.resize(pic1, (31, 31), interpolation=cv2.INTER_CUBIC)
        # pic2_resize = cv2.resize(pic2, (31, 31), interpolation=cv2.INTER_CUBIC)
        # pic3_resize = cv2.resize(pic3, (31, 31), interpolation=cv2.INTER_CUBIC)
        # pic4_resize = cv2.resize(pic4, (31, 31), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_0.png'), pic1_resize)
        # cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_1.png'), pic2_resize)
        # cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_2.png'), pic3_resize)
        # cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_3.png'), pic4_resize)
        pic1 = raw_image[idx[0] - 1 if idx[0] - 1 > 0 else 0:idx[0] + 2, idx[1] - 1 if idx[1] - 1 > 0 else 0:idx[1] + 2]
        pic2 = raw_image[idx[0] - 2 if idx[0] - 2 > 0 else 0:idx[0] + 3, idx[1] - 2 if idx[1] - 2 > 0 else 0:idx[1] + 3]
        pic3 = raw_image[idx[0] - 3 if idx[0] - 3 > 0 else 0:idx[0] + 4, idx[1] - 3 if idx[1] - 3 > 0 else 0:idx[1] + 4]
        pic4 = raw_image[idx[0] - 4 if idx[0] - 4 > 0 else 0:idx[0] + 5, idx[1] - 4 if idx[1] - 4 > 0 else 0:idx[1] + 5]
        pic5 = raw_image[idx[0] - 5 if idx[0] - 5 > 0 else 0:idx[0] + 6, idx[1] - 5 if idx[1] - 5 > 0 else 0:idx[1] + 6]
        pic6 = raw_image[idx[0] - 6 if idx[0] - 6 > 0 else 0:idx[0] + 7, idx[1] - 6 if idx[1] - 6 > 0 else 0:idx[1] + 7]
        pic7 = raw_image[idx[0] - 7 if idx[0] - 7 > 0 else 0:idx[0] + 8, idx[1] - 7 if idx[1] - 7 > 0 else 0:idx[1] + 8]
        pic8 = raw_image[idx[0] - 8 if idx[0] - 8 > 0 else 0:idx[0] + 9, idx[1] - 8 if idx[1] - 8 > 0 else 0:idx[1] + 9]
        pic9 = raw_image[idx[0] - 9 if idx[0] - 9 > 0 else 0:idx[0] + 10,
               idx[1] - 9 if idx[1] - 9 > 0 else 0:idx[1] + 10]
        pic10 = raw_image[idx[0] - 10 if idx[0] - 10 > 0 else 0:idx[0] + 11,
                idx[1] - 10 if idx[1] - 10 > 0 else 0:idx[1] + 11]
        pic11 = raw_image[idx[0] - 11 if idx[0] - 11 > 0 else 0:idx[0] + 12,
                idx[1] - 11 if idx[1] - 11 > 0 else 0:idx[1] + 12]
        pic12 = raw_image[idx[0] - 12 if idx[0] - 12 > 0 else 0:idx[0] + 13,
                idx[1] - 12 if idx[1] - 12 > 0 else 0:idx[1] + 13]
        pic13 = raw_image[idx[0] - 13 if idx[0] - 13 > 0 else 0:idx[0] + 14,
                idx[1] - 13 if idx[1] - 13 > 0 else 0:idx[1] + 14]
        pic14 = raw_image[idx[0] - 14 if idx[0] - 14 > 0 else 0:idx[0] + 15,
                idx[1] - 14 if idx[1] - 14 > 0 else 0:idx[1] + 15]
        pic15 = raw_image[idx[0] - 15 if idx[0] - 15 > 0 else 0:idx[0] + 16,
                idx[1] - 15 if idx[1] - 15 > 0 else 0:idx[1] + 16]
        pic1_resize = cv2.resize(pic1, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic2_resize = cv2.resize(pic2, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic3_resize = cv2.resize(pic3, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic4_resize = cv2.resize(pic4, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic5_resize = cv2.resize(pic5, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic6_resize = cv2.resize(pic6, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic7_resize = cv2.resize(pic7, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic8_resize = cv2.resize(pic8, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic9_resize = cv2.resize(pic9, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic10_resize = cv2.resize(pic10, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic11_resize = cv2.resize(pic11, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic12_resize = cv2.resize(pic12, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic13_resize = cv2.resize(pic13, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic14_resize = cv2.resize(pic14, (31, 31), interpolation=cv2.INTER_CUBIC)
        pic15_resize = cv2.resize(pic15, (31, 31), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_00.png'), pic1_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_01.png'), pic2_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_02.png'), pic3_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_03.png'), pic4_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_04.png'), pic5_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_05.png'), pic6_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_06.png'), pic7_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_07.png'), pic8_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_08.png'), pic9_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_09.png'), pic10_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_10.png'), pic11_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_11.png'), pic12_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_12.png'), pic13_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_13.png'), pic14_resize)
        cv2.imwrite(os.path.join(val_dir, str(i).zfill(5) + '_14.png'), pic15_resize)
        i += 1


if __name__ == '__main__':
    raw_image = cv2.imread('./HSIdata/rawdata/ip/indian_pca.jpg')
    gt_data = loadmat('./HSIdata/rawdata/ip/Indian_pines_gt.mat')['indian_pines_gt']
    train_dir = './HSIdata/ip_50p_17c/train/'
    val_dir = './HSIdata/ip_50p_17c/val/'

    for cls in range(17):
        print('====> processing cls {}'.format(str(cls)))
        train_dir1 = os.path.join(train_dir, str(cls))
        val_dir1 = os.path.join(val_dir, str(cls))
        if not os.path.exists(train_dir1):
            os.makedirs(train_dir1)
        if not os.path.exists(val_dir1):
            os.makedirs(val_dir1)

        index = np.argwhere(gt_data == cls)
        # print(index[0:5])
        train_num = math.ceil(len(index) * 0.5)
        # train_num = 5
        val_num = int(len(index) - train_num)
        index_choice = np.random.choice(np.arange(len(index)), train_num + val_num, replace=False)
        index_choice = index[index_choice]
        train_choice = index_choice[0: train_num]
        val_choice = index_choice[train_num: train_num+val_num]

        # make train/val/test dataset
        getTrainimage(raw_image, train_choice, train_dir1)
        if cls != 0:
            getValimage(raw_image, val_choice, val_dir1)
            # getValimage(raw_image, index, val_dir1)
        lbl = np.zeros(17)
        lbl[cls] += 1
        # # i += 1
        savemat(train_dir1 + '/label.mat', {'label': lbl})
        savemat(val_dir1 + '/label.mat', {'label': lbl})

