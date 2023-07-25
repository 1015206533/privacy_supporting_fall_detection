import mmcv
from mmcv.fileio import FileClient
import os
import cv2
import sys
from PIL import Image as image_read
import matplotlib.image as mpimg 
import numpy as np
import matplotlib.pyplot as plt

def analysis(filename):
    output_directory = os.path.dirname(filename)  # 提取文件的路径
    output_name = filename.split('.')[0]  # 提取文件名

    # file_client_d = FileClient('disk', **{})
    # img_bytes_d = file_client_d.get(filename)
    # arr = mmcv.imfrombytes(img_bytes_d, channel_order='rgb')

    arr = cv2.imread(filename, -1)

    # arr = arr.reshape(arr.shape[:2])
    # arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))*65535
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    # arr = arr.astype(np.uint16)
    print(arr.shape)
    print(np.min(arr))
    print(np.max(arr))
    print(arr)
    img_path = os.path.join(output_directory, "{}_disp.png".format(output_name))
    cv2.imwrite(img_path, arr)

if "__main__" == __name__:
    filename= sys.argv[1]
    analysis(filename)