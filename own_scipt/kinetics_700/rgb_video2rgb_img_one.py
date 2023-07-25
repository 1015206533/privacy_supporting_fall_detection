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
    video_path = os.path.join(path, 'falling_off_chair')
    target_image_path = os.path.join(path, 'rgb_falling_off_chair')
    threads = []
    thd = threading.Thread(target=task, args=('sKZmMJkSzbY_000000_000010.mp4', target_image_path, video_path, 135))
    thd.start()
    threads.append(thd)
    time.sleep(0.01)
    for it in threads:
        it.join()


def main(argv):
	for dirs in argv[1:]:
		process(dirs)

if __name__=='__main__':
	argv = sys.argv
	main(argv)

