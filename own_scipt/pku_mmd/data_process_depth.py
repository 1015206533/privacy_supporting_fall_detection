import os
import sys
import cv2
import numpy as np
import shutil

def process(path):
    sample_info_file = os.path.join(path, 'rgb_sample_info.txt')
    target_depth = os.path.join(path, 'depth_sample')
    if not os.path.exists(target_depth):
        os.makedirs(target_depth)
    sample_info_list = []
    with open(sample_info_file, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            if len(info) != 5:
                continue
            sample_info_list.append(info)
    
    depth_path = os.path.join(path, 'DEPTH_v2')
    for sample in sample_info_list:
        file_type = 'data-' + sample[0].split('_')[0].split('-')[1] + '-depth'
        source_path = os.path.join(depth_path, file_type, sample[0].split('_')[0], 'depth')
        st, ed = max(int(sample[3]), 0), int(sample[4])
        target_path_sample = ''
        if sample[1] == '1':
            target_path_sample = os.path.join(target_depth, sample[0].split('_')[0]  + '_depth_fall')
        else:
            target_path_sample = os.path.join(target_depth, sample[0].split('_')[0]  + '_depth_adl')
        if not os.path.exists(target_path_sample):
            os.makedirs(target_path_sample)
        for i in range(st, ed, 1):
            source_file = os.path.join(source_path, str(i)+'.png')
            if not os.path.exists(source_file):
                continue
            target_flie = os.path.join(target_path_sample, str(i-st+1).zfill(4)+'.png')
            shutil.copyfile(source_file, target_flie)


def main(argv):
	for dirs in argv[1:]:
		process(dirs)

if __name__=='__main__':
	argv = sys.argv
	main(argv)
