import os
import sys
import cv2
import time
import numpy as np
import threading


def generate_video_and_images(video_path, target_image_path, video_file, index):
    target_image_name = '{:04}'.format(index+1) + '_' + video_file.split('_')[0]
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


def task(file, target_image_path, video_path, index):
    generate_video_and_images(video_path, target_image_path, file, index)
    print(index, file)
    return 


def process(path):
    video_path = os.path.join(path, 'sweeping_floor')
    target_image_path = os.path.join(path, 'rgb_sweeping_floor')
    if not os.path.exists(target_image_path):
        os.makedirs(target_image_path)
    file_list = os.listdir(video_path)
    threads = []
    for i, file in enumerate(file_list):
        if i != 278 and i < 673:
            continue
        if i >= 746:
            break
        thd = threading.Thread(target=task, args=(file, target_image_path, video_path, i))
        thd.start()
        threads.append(thd)
        time.sleep(0.1)
    for it in threads:
        it.join()


def main(argv):
	for dirs in argv[1:]:
		process(dirs)

if __name__=='__main__':
	argv = sys.argv
	main(argv)

