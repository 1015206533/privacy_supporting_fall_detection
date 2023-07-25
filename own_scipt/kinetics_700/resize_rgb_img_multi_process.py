'''
使用多进程批量处理图片
'''
from multiprocessing.dummy import Pool
from PIL import Image
import mmcv
import numpy as np
import os
import sys 
import cv2
from PIL import Image

global count
count=1

def get_image_paths(input_path, output_path):
    file_list = os.listdir(input_path)
    in_list = []
    for file_name in file_list:
        input_path_new = os.path.join(input_path, file_name)
        output_path_new = os.path.join(output_path, file_name)
        if os.path.isdir(input_path_new):
            if not os.path.exists(output_path_new):
                os.makedirs(output_path_new)
            in_list_tmp = get_image_paths(input_path_new, output_path_new)
            in_list.extend(in_list_tmp)
        else:
            if not os.path.exists(output_path_new):
                in_list.append([input_path_new, output_path_new])
    print(len(in_list))
    return in_list


def resize_image(filename):
    global count
    input, output = filename
    try:
        img = cv2.imread(input)
        img_resize = mmcv.imresize(img, (256, 256), interpolation='bilinear')
        cv2.imwrite(output, img_resize)
        if count%1000 == 0:
            print('存储resize之后的图像，第 {} 张，存储成功！'.format(count))
        count = count+1
    except:
        print('error: ', input)


def main(input_path, output_path):
    in_list = get_image_paths(input_path, output_path)
    pool = Pool(24)
    pool.map(resize_image, in_list)  # 注意map用法，是multiprocessing.dummy.Pool的方法
    pool.close()
    pool.join()


if __name__=='__main__':
	argv = sys.argv
	main(argv[1], argv[2])
