import os
import sys
import cv2
import time
import numpy as np
import threading


def generate_video_and_images(video_path, target_image_path, video_file, label):
    target_image_name = ''
    if label == 1:
        target_image_name = video_file.split('_')[0] + '_rgb_fall'
    else:
        target_image_name = video_file.split('_')[0] + '_rgb_adl'
    target_image = os.path.join(target_image_path, target_image_name)
    if not os.path.exists(target_image):
        os.makedirs(target_image)
    
    source_video_file = os.path.join(video_path, video_file)
    cap = cv2.VideoCapture(source_video_file)
    _, frame = cap.read()
    i = 0
    
    while frame is not None:
        image_file = os.path.join(target_image, 'img_{:04}.png'.format(i+1))
        cv2.imwrite(image_file,frame)
        _, frame = cap.read()
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return 


def task(file, target_image_path, video_path, sample_info_file):
    if file.split('_')[0][-4:] != 'A043':
        return
    adl_index = np.random.randint(1, 61)
    while adl_index == 43:
        adl_index = np.random.randint(1, 61)
    adl_file = file.split('_')[0][:-3] + '{:03}'.format(adl_index) + '_rgb.avi'

    generate_video_and_images(video_path, target_image_path, file, 1)
    generate_video_and_images(video_path, target_image_path, adl_file, 0)
    print('\t'.join(map(str, [file, 1])))
    sample_info_file.write('\t'.join(map(str, [file, 1])) + '\n')
    print('\t'.join(map(str, [adl_file, 0])))
    sample_info_file.write('\t'.join(map(str, [adl_file, 0])) + '\n')
    return 


def process(path):
    sample_info_file_name = os.path.join(path, 'rgb_sample_info.txt')
    sample_info_file = open(sample_info_file_name, 'w')
    video_path = os.path.join(path, 'rgb', 'nturgb+d_rgb')
    target_image_path = os.path.join(path, 'rgb_sample')
    if not os.path.exists(target_image_path):
        os.makedirs(target_image_path)
    file_list = os.listdir(video_path)
    threads = []
    for file in file_list:
        thd = threading.Thread(target=task, args=(file, target_image_path, video_path, sample_info_file))
        thd.start()
        threads.append(thd)
        time.sleep(0.01)
    for it in threads:
        it.join()
    sample_info_file.close()


def main(argv):
	for dirs in argv[1:]:
		process(dirs)

if __name__=='__main__':
	argv = sys.argv
	main(argv)

