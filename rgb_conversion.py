import os
import cv2
import numpy as np
from config import *

#CNN1, 2, 3のテンソルを作る関数
def make_cut_imgae_array(img, place):
    # make empty ndarray
    img1_array = np.empty((0, 4096), int)
    # when CNN1 image
    if place == 1:
        cut_img = img[0:64, 32:96]
    # when CNN1 image
    elif place == 2:
        cut_img = img[64:128, 0:64]
    # when CNN2 image
    else:
        cut_img = img[64:128, 64:128]
    return np.reshape(cut_img, (1, 4096, 3))

def make_imgae_array(dir):
    # make empty ndarray for return
    data_array = np.empty((0,16384,3), int)
    data_array_1 = np.empty((0, 4096, 3), int)
    data_array_2 = np.empty((0, 4096, 3), int)
    data_array_3 = np.empty((0, 4096, 3), int)
    for i, img_name in enumerate(dir):
        print(img_name)
        # make empty ndarray for CNN0 image
        img_array = np.empty((0, 16384), int)
        # read image
        img = cv2.imread(os.path.join(RAW_DIR, img_name))
        # get image from left side
        cut_img = img[0:1280, 0:1280]
        # reshape 1280x1280 to 128x128pxl
        resized_img = cv2.resize(cut_img, (128, 128))
        # reshaped image to 1x16384x3 ndarray
        reshaped_img = np.reshape(resized_img, (1, 16384, 3))
        # cut and reshape 128x128 image to 1x4096x3
        reshaped_img1 = make_cut_imgae_array(resized_img, 1)
        reshaped_img2 = make_cut_imgae_array(resized_img, 2)
        reshaped_img3 = make_cut_imgae_array(resized_img, 3)
        # append to ndarray that for return
        data_array = np.append(data_array, reshaped_img, axis=0)
        data_array_1 = np.append(data_array_1, reshaped_img1, axis=0)
        data_array_2 = np.append(data_array_2, reshaped_img2, axis=0)
        data_array_3 = np.append(data_array_3, reshaped_img3, axis=0)
    return data_array, data_array_1, data_array_2, data_array_3

'''
返り値 4種類
0 : CNN0用のテンソル
1 : CNN1用のテンソル
2 : CNN2用のテンソル
3 : CNN3用のテンソル
data0, data1, data2, data3 = make_imgae_array(os.listdir(RAW_DIR))
'''
