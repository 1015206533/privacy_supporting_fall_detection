import os
import sys
import cv2
import time
import numpy as np
import threading
import mmcv
from mmcv.fileio import FileClient
from PIL import Image as image_read
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt


def task(file_path, file_name, output_path):
    path_file = os.path.join(file_path, file_name)
    output_file = os.path.join(output_path, file_name)
    img = cv2.imread(path_file)
    img_resize = mmcv.imresize(img, (256, 256), interpolation='bilinear')
    cv2.imwrite(output_file, img_resize)


def do_resize(file_path, file_name, output_path):
    path_file = os.path.join(file_path, file_name)
    output_path_new = os.path.join(output_path, file_name)
    if not os.path.exists(output_path_new):
        os.makedirs(output_path_new)
    file_list = os.listdir(path_file)
    for file_name_new in file_list:
        task(path_file, file_name_new, output_path_new)
    

def main(input_path, output_path):
    file_list = os.listdir(input_path)
    threads = []
    for file_name in file_list:
        thd = threading.Thread(target=do_resize, args=(input_path, file_name, output_path))
        thd.start()
        threads.append(thd)
    for it in threads:
        it.join()


if __name__=='__main__':
	argv = sys.argv
	main(argv[1], argv[2])





