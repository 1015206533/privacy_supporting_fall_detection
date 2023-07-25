import os
import sys
import cv2
import time
import numpy as np
import threading

def generate_video_and_images(video_path, video_file, st, ed, label):
    target_video_name = ''
    target_image_name = ''
    if label == 1:
        target_video_name = video_file.split('_')[0] + '_rgb_fall.avi'
        target_image_name = video_file.split('_')[0] + '_rgb_fall'
    else:
        target_video_name = video_file.split('_')[0] + '_rgb_adl.avi'
        target_image_name = video_file.split('_')[0] + '_rgb_adl'
    target_video = os.path.join(video_path, target_video_name)
    target_image = os.path.join(video_path, target_image_name)
    if not os.path.exists(target_image):
        os.makedirs(target_image)
    
    source_video_file = os.path.join(video_path, video_file)
    cap = cv2.VideoCapture(source_video_file)
    _, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(target_video, fourcc, cap.get(5), (frame.shape[1], frame.shape[0]))
    i = 0
    while frame is not None:
        if i >= st and i < ed:
            out.write(frame)
            image_file = os.path.join(target_image, str(i+1-st).zfill(4)+'.png')
            cv2.imwrite(image_file,frame)
        _, frame = cap.read()
        i += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return 


def task(file, label_path, video_path, sample_info_file):
    name_head = file.split('_')[0]
    label_name = name_head + '.txt'
    label_file = os.path.join(label_path, label_name)
    if not os.path.exists(label_file):
        print('file not exists: ', label_file)
        return
    has_fall = False
    fall_index = -1
    action_list = []
    max_frame_num = 0
    with open(label_file, 'r') as f:
        i = 0
        for line in f:
            info = line.strip().split(',')
            if len(info) < 4:
                continue
            max_frame_num = max(max_frame_num, int(info[1]), int(info[2]))
            action_list.append([info[0], info[1], info[2]])
            if info[0] == '11':
                has_fall = True
                fall_index = i 
            i += 1
    if has_fall:
        fall_info = action_list[fall_index]
        pos_st, pos_ed = int(fall_info[1])-50, int(fall_info[2])+50
        adl_index = 0
        times = 0
        while True:
            adl_index = np.random.randint(0, len(action_list))
            st, ed = int(action_list[adl_index][1])-50, int(action_list[adl_index][2])+50
            flag = True
            if st >= pos_st and st <= pos_ed:
                flag = False
            if ed >= pos_st and ed <= pos_ed:
                flag = False
            if st < 0 or ed > max_frame_num:
                flag = False
            if flag:
                break
            times += 1
            if times > 1000:
                break
        if times > 1000:
            return
        adl_info = action_list[adl_index]
        neg_st, neg_ed = int(adl_info[1])-50, int(adl_info[2])+50
        print(file, fall_info[0], pos_st, pos_ed, adl_info[0], neg_st, neg_ed)
        generate_video_and_images(video_path, file, pos_st, pos_ed, 1)
        generate_video_and_images(video_path, file, neg_st, neg_ed, 0)
        sample_info_file.write('\t'.join(map(str, [file, 1, fall_info[0], pos_st, pos_ed])) + '\n')
        sample_info_file.write('\t'.join(map(str, [file, 0, adl_info[0], neg_st, neg_ed])) + '\n')
    return 


def process(path):
    sample_info_file_name = os.path.join(path, 'rgb_sample_info.txt')
    sample_info_file = open(sample_info_file_name, 'w')
    label_path = os.path.join(path, 'Label')
    video_path = os.path.join(path, 'RGB_VIDEO_v2')
    file_list = os.listdir(video_path)
    threads = []
    for file in file_list:
        thd = threading.Thread(target=task, args=(file, label_path, video_path, sample_info_file))
        thd.start()
        threads.append(thd)
        time.sleep(2)
    for it in threads:
        it.join()
    sample_info_file.close()


def main(argv):
	for dirs in argv[1:]:
		process(dirs)

if __name__=='__main__':
	argv = sys.argv
	main(argv)

