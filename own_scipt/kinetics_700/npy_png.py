
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

def convert(filename):
    output_directory = os.path.dirname(filename)  # 提取文件的路径
    output_name = filename.split('.')[0]  # 提取文件名
    arr = np.load(filename)  # 提取 npy 文件中的数组
    arr = arr.reshape(arr.shape[2:])
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))*65535
    arr = arr.astype(np.uint16)
    print(arr.shape)
    print(np.min(arr))
    print(np.max(arr))
    print(arr)
    img_path = os.path.join(output_directory, "{}_disp.png".format(output_name))
    cv2.imwrite(img_path, arr)
    # disp_to_img = scipy.misc.imresize( arr , [arr.shape[0], arr.shape[1]])  # 根据 需要的尺寸进行修改
    # plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), arr, cmap='plasma')  # 定义命名规则，保存图片


if "__main__" == __name__:
    filename= sys.argv[1]
    convert(filename)
