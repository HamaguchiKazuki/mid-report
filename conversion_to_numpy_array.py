import os
import cv2
import numpy as np
from config import *

def make_image_array(dir):
    data_array = np.empty((0,16384), int)
    for i, img_name in enumerate(dir):
        img = cv2.imread(os.path.join(RAW_DIR, img_name))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cut_img = gray_img[0:1280, 0:1280]
        resized_0 = cv2.resize(cut_img, (128, 128))
        resized = np.reshape(resized_0, (1, 16384))
        data_array = np.append(data_array, resized, axis=0)
    return data_array
